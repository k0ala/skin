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
    num_threads=6, shuffle=True):
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
            capacity = batch_size*(num_threads+2),
            shuffle = shuffle
        )

        images = tf.read_file(filenames)
        images = tf.image.decode_jpeg(images, channels=3)

        if base_net == 'inception_v1': image_size = 224
        if base_net == 'inception_v3': image_size = 299

        if distort:
            images = tf.image.resize_images(images, image_size/5*6, image_size/5*6)
            tf.image_summary('orig_image', tf.expand_dims(images, 0))
            images = tf.image.random_ops.random_crop(images, [image_size]*2+[3])
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

        if shuffle:
            images, labels, filenames = tf.train.shuffle_batch(
                tensor_list = [images, labels, filenames],
                batch_size = batch_size,
                num_threads = num_threads,
                capacity = batch_size*(num_threads+2),
                min_after_dequeue = batch_size
            )
        else:
            images, labels, filenames = tf.train.batch(
                tensor_list = [images, labels, filenames],
                batch_size = batch_size,
                num_threads = num_threads,
                capacity = batch_size*(num_threads+2),
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


def normalize_batch(batch, phase, beta=None, gamma=None):

    d = batch.get_shape().as_list()[-1]
    r = len(batch.get_shape())
        
    if beta is None:
        beta = tf.Variable(tf.zeros([d]), name='beta')
    if gamma is None:
        gamma = tf.Variable(tf.ones([d]), name='gamma')

    mu, var = tf.nn.moments(batch, axes=range(r-1))

    if phase == 'test':
        mu_name = mu.op.name + '/ExponentialMovingAverage'
        var_name = var.op.name + '/ExponentialMovingAverage'
        del mu, var
        mu, var = tf.zeros([d]), tf.ones([d])
        with tf.name_scope(''):
            mu = tf.Variable(mu, name=mu_name)
            var = tf.Variable(var, name=var_name)

    if phase == 'train':
        tf.add_to_collection('batch_moments', mu)
        tf.add_to_collection('batch_moments', var)
            
    if r is 4: 
        batch = tf.nn.batch_norm_with_global_normalization(
            t=batch, m=mu, v=var, beta=beta, gamma=gamma, 
            variance_epsilon=1e-4, scale_after_normalization=True)
    elif r is 2: 
        batch -= mu
        batch *= tf.rsqrt(var+1e-4)
        batch *= gamma
        batch += beta

    return batch


def _apply_op(op, tensors):

    op = tf.get_default_graph().create_op(
        op_type = op.type, 
        inputs = [tf.convert_to_tensor(tensors[i.op.name]) for i in op.inputs],
        dtypes = [i.dtype for i in op.outputs], 
        name = op.name, 
        attrs =  op.node_def.attr
    )

    return op.outputs[0]


def _get_features_v1(images, graph, phase, batch_normalize):
    """Compute features for inception_v1 net

    Args:
        images (tensorflow.Tensor): 4D array of image batch,
            batch_size by image dimensions
        graph (tensorflow.Graph): Graph with imagenet trained parameters
        batch_normalize (bool): Whether to use batch normalization

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

    config = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=config)
    with tf.Session(graph=graph, config=config) as sess:
        weights_orig = {op.name: op.outputs[0].eval() for op in weights_orig}
        bias_orig = {op.name: op.outputs[0].eval() for op in bias_orig}

    tensors_new = {
        name: tf.Variable(
            initial_value = value,
            name = name[:-2] + '_pre_relu/weights',
            collections = ['variables', 'weights']
        )
        for name, value in weights_orig.iteritems()
    }

    suffix = '_pre_relu/' + ['bias', 'batchnorm/beta'][batch_normalize]
    
    tensors_new.update({
        name: tf.Variable(value, name=name[:-2]+suffix)
        for name, value in bias_orig.iteritems()
    })

    tensors_new[input_op.name] = images

    for op in reuse_ops:

        if batch_normalize and op.type == 'BiasAdd':
            batch, bias = [tensors_new[i.op.name] for i in op.inputs]
            with tf.name_scope(op.name + '/batchnorm/'):
                batch = normalize_batch(batch, phase, beta=bias)
            tensors_new[op.name] = tf.identity(batch, name=op.name)

        else:
            tensors_new[op.name] = _apply_op(op, tensors_new)

    features = [tensors_new[i] for i in ['nn0', 'nn1', 'avgpool0/reshape']]
    features = [tf.identity(i, name='features') for i in features]

    return features


def _get_features_v3(images, graph, phase):
    """Compute features for inception_v3 net

    Args:
        images (tensorflow.Tensor): 4D array of image batch,
            batch_size by image dimensions
        graph (tensorflow.Graph): Graph with imagenet trained parameters

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

    config = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=config)
    with tf.Session(graph=graph, config=config) as sess:
        weights_orig = {op.name: op.outputs[0].eval() for op in weights_ops}
        bias_orig = {op.name: op.outputs[0].eval() for op in bias_orig}

    tensors_new = {
        name: tf.Variable(
            initial_value = value,
            name = name,
            collections = ['variables', 'weights']
        )
        for name, value in weights_orig.iteritems()
    }

    tensors_new.update({
        name: tf.Variable(value, name=name)
        for name, value in bias_orig.iteritems()
    })

    tensors_new[input_op.name] = images

    for op in reuse_ops:

        if op.type == 'BatchNormWithGlobalNormalization':
            batch, beta, gamma = [
                tensors_new[op.inputs[i].op.name] for i in [0,3,4]
            ]
            with tf.name_scope(op.name + '/'):
                batch = normalize_batch(batch, phase, beta, gamma)
            tensors_new[op.name] = tf.identity(batch, name=op.name)

        else:
            tensors_new[op.name] = _apply_op(op, tensors_new)

    features = tf.reshape(tensors_new[features_op.name], [-1, 2048], name='features')

    return features


def get_features(images, base_net, phase, batch_normalize):
    """Compute features

    Args:
        images (tensorflow.Tensor): 4D array of image batch,
            batch_size by image dimensions
        base_net (string): Net to finetune, either inception_v1 or inception_v3
        batch_normalize (bool): Whether to use batch normalization

    Returns:
        features (tensorflow.Tensor): 2D array of features,
        batch size by features dimension
    """

    graph = tf.Graph()

    graph_def = tf.GraphDef()
    graph_def_file = os.path.join('../graphs/', base_net+'.pb')
    with open(graph_def_file) as f: graph_def.ParseFromString(f.read())
    with graph.as_default(): tf.import_graph_def(graph_def, name='')

    if base_net == 'inception_v1':
        features = _get_features_v1(images, graph, phase, batch_normalize)

    if base_net == 'inception_v3':
        features = _get_features_v3(images, graph, phase)

    return features

def build_graph(hypes, filenames, labels, num_classes, phase, batch_size):

    images, labels, filenames = get_inputs(
        filenames = filenames,
        labels = labels, 
        batch_size = batch_size,
        base_net = hypes['base_net'],
        num_threads = 1 + 5*(phase=='train'),
        distort = (phase=='train'),
        shuffle = (phase=='train')
    )

    features = get_features(
        images = images,
        base_net = hypes['base_net'],
        phase = phase,
        batch_normalize = hypes['batch_normalize']
    )

    if type(features) is list:

        loss = []
        feats = features[-1]

        for i, features in enumerate(features):
            
            tf.histogram_summary(features.op.name, features)

            if hypes['use_dropout'] and phase == 'train':
                features = tf.nn.dropout(features, 0.5)

            with tf.name_scope('logits_{}'.format(i)):
                logits = get_logits(
                    features = features,
                    num_classes = num_classes
                )

            with tf.name_scope('loss_{}'.format(i)):
                loss.append(get_loss(
                    logits = logits, 
                    labels = labels,
                    loss_matrix=hypes['loss_matrix']
                ))

        with tf.name_scope('loss'):
            loss = tf.reduce_sum(
                input_tensor = tf.mul(tf.pack(loss), hypes['head_weights']),
                name = 'loss'
            )

    else:
        
        feats = features
        if hypes['use_dropout'] and phase == 'train':
            features = tf.nn.dropout(features, 0.5)

        with tf.name_scope('logits'):
            logits = get_logits(
                features = features,
                num_classes = num_classes
            )

        with tf.name_scope('loss'):
            loss = get_loss(
                logits = logits, 
                labels = labels,
                loss_matrix = hypes['loss_matrix']
            )

    fetches = {
        'loss': loss,
        'filenames': filenames,
        'logits': logits,
        'features': feats
    }

    if phase == 'test':
        for op in tf.get_default_graph().get_operations():
            print op.type.ljust(35), '\t', op.name

    if phase == 'train':

        with tf.variable_scope('weights_norm') as scope:
            weights_norm = tf.reduce_sum(
                input_tensor = tf.pack(
                    [tf.nn.l2_loss(i) for i in tf.get_collection('weights')]
                ),
                name='weights_norm'
            )
        
        with tf.variable_scope('regularized_loss'):
            regularized_loss = tf.add(
                x = loss, 
                y = hypes['weight_decay']*weights_norm, 
                name = 'regularized_loss'
            )

        for op in tf.get_default_graph().get_operations():
            print op.type.ljust(35), '\t', op.name

        global_step = tf.Variable(0, name='global_step', trainable=False)

        opt = eval('tf.train.{}Optimizer'.format(hypes['optimizer']))(
            learning_rate = hypes['learning_rate'], 
            epsilon = hypes['epsilon']
        )

        grads = opt.compute_gradients(regularized_loss)
        apply_grads = opt.apply_gradients(
            grads_and_vars = grads, 
            global_step = global_step
        )

        if hypes['batch_normalize']:
            moving_average = tf.train.ExponentialMovingAverage(0.99)
            batch_moments = tf.get_collection('batch_moments')
            with tf.control_dependencies([apply_grads]):
                move_averages = moving_average.apply(batch_moments)
                train_op = move_averages
        else:
            train_op = apply_grads

        tf.histogram_summary(logits.op.name, logits)
        tf.scalar_summary(loss.op.name, loss)
        tf.scalar_summary(weights_norm.op.name, weights_norm)
        for grad, var in grads:
            tf.histogram_summary(var.op.name, var)
            tf.histogram_summary(var.op.name + '/gradients', grad)

        fetches.update({
            'weights_norm': weights_norm,
            'train_op': train_op,
            'global_step': global_step
        })

    return fetches