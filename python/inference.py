"""Builds the network.

Summary of available functions:

    - Load image and label batches
    - Compute features
    - Compute logits
    - Compute loss

"""

import os
import numpy as np
import tensorflow as tf


def get_inputs(filenames, labels, batch_size, base_net, distort=True, 
    num_threads=6):
    """Load image and label batches

    Args:
        filenames (list): List of filename paths
        labels (list): List of label integers
        batch_size (int): Number of images in a batch
        base_net (string): Net to finetune, either inception_v1 or inception_v3
        distort (bool): If `True`, randomly flip and crop the image
        num_threads (int): Number of threads to use for loading image batches

    Returns:
        images, labels, filenames (tuple): Image, label, and filename batches
    """

    with tf.name_scope('inputs'):

        filenames, labels = tf.train.slice_input_producer(
            tensor_list = [filenames, labels],
            capacity = batch_size*(num_threads+2)
        )

        images = tf.read_file(filenames)
        images = tf.image.decode_jpeg(images, channels=3)

        if base_net == 'inception_v1': image_size = 224
        if base_net == 'inception_v3': image_size = 299

        if distort:
            images = tf.image.resize_images(images, image_size/5*6, image_size/5*6)
            tf.image_summary('orig_image', tf.expand_dims(images, 0))
            images = tf.image.random_crop(images, [image_size]*2)
            images = tf.image.random_flip_left_right(images)
            images = tf.image.random_flip_up_down(images)
            tf.image_summary('train_image', tf.expand_dims(images, 0))
        else:
            images = tf.image.resize_images(images, image_size, image_size)

        if base_net == 'inception_v1':
            images -= 117

        if base_net == 'inception_v3':
            images -= 128
            images /= 128

        images, labels, filenames = tf.train.shuffle_batch(
            tensor_list = [images, labels, filenames],
            batch_size = batch_size,
            num_threads = num_threads,
            capacity = batch_size*(num_threads+2),
            min_after_dequeue = batch_size
        )

    return images, labels, filenames


def get_logits(features, num_classes):
    """Compute logits

    Args:
        features (tensorflow.Tensor): 2D array of features,
            batch size by features dimension
        num_classes (int): Number of classes

    Returns:
        logits (tensorflow.Tensor): 2D array of logits, 
        batch size by number of classes
    """

    features_dim = features.get_shape().as_list()[-1]

    w = tf.Variable(
        initial_value = (
            tf.random_normal(
                shape = [features_dim, num_classes],
                stddev = 1e-4,
                name = 'weights_init'
            )
        ), 
        name = 'weights',
        collections = ['variables', 'weights']
    )

    b = tf.Variable(
        initial_value = (
            tf.random_normal(
                shape = [num_classes],
                stddev = 1e-4,
                name = 'bias_init'
            )
        ), 
        name='biases',
        collections = ['variables', 'weights']
    ) 

    logits = tf.nn.xw_plus_b(features, w, b, name='logits')

    return logits


def get_loss(logits, labels, loss_matrix=None):
    """Compute loss

    Args:
        logits (tensorflow.Tensor): 2D array of logits, 
            batch size by number of classes
        labels (tensorflow.Tensor): 1D array of label integers
        loss_matrix (list): 2D array loss matrix (experimental)

    Returns:
        loss (tensorflow.Tensor): 0D array (scalar) average loss over batch
    """

    if loss_matrix:

        loss_matrix = np.array(loss_matrix, dtype='float32')
        loss_weights = tf.nn.embedding_lookup(loss_matrix, labels)
        loss = -tf.log(tf.nn.softmax(logits)+1e-8)
        loss = tf.reduce_sum(loss*loss_weights, axis=1)

    else:

        batch_size, num_classes = logits.get_shape().as_list()

        labels_sparse = tf.sparse_to_dense(
            sparse_indices = tf.transpose(
                tf.pack([tf.range(batch_size), labels])
            ),
            output_shape = [batch_size, num_classes],
            sparse_values = np.ones(batch_size, dtype='float32')
        )

        loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels_sparse)

    loss = tf.reduce_mean(loss, name='loss')

    return loss


def _get_features_v1(images, graph, batch_normalize, sess_config):
    """Compute features for inception_v1 net

    Args:
        images (tensorflow.Tensor): 4D array of image batch,
            batch_size by image dimensions
        graph (tensorflow.Graph): Graph with imagenet trained parameters
        batch_normalize (bool): Whether to use batch normalization
        sess_config (tensorflow.ConfigProto): Session configuration,
            mostly for maintaining control over GPU memory usage

    Returns:
        features (tensorflow.Tensor): 2D array of features,
        batch size by features dimension
    """

    input_op = graph.get_operation_by_name('input')

    ops = [
        op for op in graph.get_operations() 
        if 'softmax' not in op.name and 'output' not in op.name
    ]
    weights_orig = [op for op in ops if op.name[-2:] == '_w']
    bias_orig = [op for op in ops if op.name[-2:] == '_b']
    reuse_ops = [
        op for op in ops if op not in weights_orig + bias_orig + [input_op]
    ]

    with tf.Session(graph=graph, config=sess_config):
        weights_orig = {
            op.name: op.outputs[0].eval() for op in weights_orig
        }
        bias_orig = {
            op.name: op.outputs[0].eval() for op in bias_orig
        }

    T = {
        name: tf.Variable(
            initial_value = value,
            name = name[:-2] + '_pre_relu/weights',
            collections = ['variables', 'weights']
        )
        for name, value in weights_orig.iteritems()
    }

    if batch_normalize:
        name_suffix = '_pre_relu/batchnorm/beta'
    else:
        name_suffix = '_pre_relu/bias'

    T.update({
        name: tf.Variable(
            initial_value = value, 
            name = name[:-2] + name_suffix
        )
        for name, value in bias_orig.iteritems()
    })

    T[input_op.name] = images

    for op in reuse_ops:

        if batch_normalize and op.type == 'BiasAdd':

            t, beta = [T[i.op.name] for i in op.inputs]

            with tf.name_scope(op.name + '/batchnorm/'):

                gamma = tf.Variable(
                    initial_value = tf.ones(
                        shape = beta.get_shape().as_list(),
                        name = 'gamma_init'
                    ),
                    name = 'gamma'
                )

                if len(t.get_shape()) == 4: 

                    mu, var = tf.nn.moments(t, axes=[0, 1, 2])

                    with tf.name_scope(''):
                        T[op.name] = tf.nn.batch_norm_with_global_normalization(
                            t=t, m=mu, v=var, beta=beta, gamma=gamma, 
                            variance_epsilon=1e-8, scale_after_normalization=True, 
                            name=op.name
                        )

                elif len(t.get_shape()) == 2: 

                    mu, var = tf.nn.moments(t, axes=[0])

                    t -= mu
                    t /= tf.sqrt(var+1e-8)
                    t *= gamma
                    t += beta

                    with tf.name_scope(''):
                        T[op.name] = tf.identity(t, name=op.name)

        else:

            copied_op = images.graph.create_op(
                op_type = op.type, 
                inputs = [tf.convert_to_tensor(T[t.op.name]) for t in op.inputs], 
                dtypes = [o.dtype for o in op.outputs], 
                name = op.name, 
                attrs =  op.node_def.attr
            )

            T[op.name] = copied_op.outputs[0]

    features = [
        tf.identity(t, name='features')
        for t in [T['nn0'], T['nn1'], T['avgpool0/reshape']]
    ]

    return features


def _get_features_v3(images, graph, sess_config):
    """Compute features for inception_v3 net

    Args:
        images (tensorflow.Tensor): 4D array of image batch,
            batch_size by image dimensions
        graph (tensorflow.Graph): Graph with imagenet trained parameters
        sess_config (tensorflow.ConfigProto): Session configuration,
            mostly for maintaining control over GPU memory usage

    Returns:
        features (tensorflow.Tensor): 2D array of features,
        batch size by features dimension
    """

    input_op = graph.get_operation_by_name('Mul')
    features_op = graph.get_operation_by_name('pool_3')

    ops = graph.get_operations()

    weights_ops = [op for op in ops if 'params' in op.name]
    bias_orig = [op for op in ops if 'beta' in op.name or 'gamma' in op.name]

    reuse_ops = graph.get_operations()
    i_input = reuse_ops.index(input_op)
    i_features = reuse_ops.index(features_op)
    reuse_ops = reuse_ops[i_input+1:i_features+1]
    reuse_ops = [
        op for op in reuse_ops 
        if op not in weights_ops + bias_orig
        and 'moving' not in op.name
    ]

    with tf.Session(graph=graph, config=sess_config):
        weights_orig = {
            op.name: op.outputs[0].eval()
            for op in weights_ops
        }
        bias_orig = {
            op.name: op.outputs[0].eval()
            for op in bias_orig
        }

    T = {
        name: tf.Variable(
            initial_value = value,
            name = name,
            collections = ['variables', 'weights']
        )
        for name, value in weights_orig.iteritems()
    }

    T.update({
        name: tf.Variable(value, name=name)
        for name, value in bias_orig.iteritems()
    })

    T[input_op.name] = images

    for op in reuse_ops:

        if op.type == 'BatchNormWithGlobalNormalization':

            t, beta, gamma = [T[op.inputs[i].op.name] for i in [0,3,4]]
            mu, var = tf.nn.moments(t, [0,1,2], name=op.name+'/')

            T[op.name] = tf.nn.batch_norm_with_global_normalization(
                t=t, m=mu, v=var, beta=beta, gamma=gamma, 
                variance_epsilon=1e-8, scale_after_normalization=True, 
                name=op.name
            )

        else:

            copied_op = images.graph.create_op(
                op_type = op.type, 
                inputs = [tf.convert_to_tensor(T[t.op.name]) for t in op.inputs],
                dtypes = [o.dtype for o in op.outputs], 
                name = op.name, 
                attrs =  op.node_def.attr
            )

            T[op.name] = copied_op.outputs[0]

    features = tf.reshape(T[features_op.name], [-1, 2048], name='features')

    return features


def get_features(images, base_net, batch_normalize, sess_config):
    """Compute features

    Args:
        images (tensorflow.Tensor): 4D array of image batch,
            batch_size by image dimensions
        base_net (string): Net to finetune, either inception_v1 or inception_v3
        batch_normalize (bool): Whether to use batch normalization
        sess_config (tensorflow.ConfigProto): Session configuration,
            mostly for maintaining control over GPU memory usage

    Returns:
        features (tensorflow.Tensor): 2D array of features,
        batch size by features dimension
    """

    graph = tf.Graph()

    graph_def = tf.GraphDef()
    graph_def_file = os.path.join('../graphs/', base_net+'.pb')
    with open(graph_def_file) as f: graph_def.MergeFromString(f.read())
    with graph.as_default(): tf.import_graph_def(graph_def, name='')

    if base_net == 'inception_v1':
        features = _get_features_v1(images, graph, batch_normalize, sess_config)

    if base_net == 'inception_v3':
        features = _get_features_v3(images, graph, sess_config)

    return features