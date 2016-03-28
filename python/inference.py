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
from inception import inception_model


def build_graph(hypes, filenames, labels, num_classes, subset, batch_size):

    with tf.name_scope('inputs'):

        filenames, labels = tf.train.slice_input_producer(
            tensor_list = [filenames, labels],
            capacity = batch_size*2,
            shuffle = (subset=='train')
        )

        filenames, labels = tf.train.batch(
            tensor_list = [filenames, labels],
            batch_size = batch_size,
            capacity = batch_size*2,
        )

        images0 = [tf.read_file(i[0]) for i in tf.split(0, batch_size, filenames)]
        images0 = [tf.image.decode_jpeg(i, channels=3) for i in images0]

        if subset == 'train':
            images0 = [tf.image.random_flip_left_right(i) for i in images0]
            images0 = [tf.image.random_flip_up_down(i) for i in images0]

        images = tf.pack([
            tf.image.resize_images(i, 299, 299) 
            for i in images0
        ])

        if hypes['spatial_transformer']:
            with tf.variable_scope('spatial_transform'):
                images = spatial_tranform(images, images0, batch_size, subset,
                    hypes['loc_net'], hypes['xform_reg'])

        with tf.name_scope(None):
            images = tf.identity(images, name='input')

    logits, logits_aux = inception_model.inference(
        images = (images-128)/128.,
        num_classes = num_classes,
        for_training = (subset == 'train'),
        restore_logits = (subset != 'train')
    )

    with tf.name_scope(None): logits = tf.identity(logits, name='logits')
    tf.histogram_summary('logits', logits)

    with tf.name_scope('loss'):

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

        loss_aux = tf.nn.softmax_cross_entropy_with_logits(logits_aux, labels_sparse)
        loss_aux = tf.reduce_mean(loss_aux, name='loss_aux')

        loss = 0.7*loss + 0.3*loss_aux

        tf.scalar_summary('loss', loss)

    fetches = {
        'loss': loss,
        'filenames': filenames,
        'logits': logits
    }

    def print_graph_ops():
        with open('/tmp/graph_ops.txt', 'w') as f:
            for op in tf.get_default_graph().get_operations():
                f.write(op.type.ljust(35)+'\t'+op.name+'\n')

    if subset == 'train':

        reg_loss = tf.add_n(tf.get_collection('regularization_losses'))
        tf.scalar_summary('reg_loss', reg_loss)

        with tf.variable_scope('reg_loss'):
            loss += reg_loss

        print_graph_ops()

        global_step = tf.Variable(0, name='global_step', trainable=False)

        opt = eval('tf.train.{}Optimizer'.format('Adam'))(
            learning_rate = hypes['learning_rate'], 
            epsilon = hypes['epsilon'],
            beta1 = hypes['beta1'],
            beta2 = hypes['beta2']
        )

        grads = opt.compute_gradients(loss)
        apply_grads = opt.apply_gradients(grads, global_step)

        variable_averages = tf.train.ExponentialMovingAverage(
            hypes['variable_averages_decay'], global_step)
        variables_to_average = (tf.trainable_variables() +
                                tf.moving_average_variables())
        variables_averages_op = variable_averages.apply(variables_to_average)

        batchnorm_updates_op = tf.group(*tf.get_collection('_update_ops_'))

        train_op = tf.group(apply_grads, variables_averages_op, batchnorm_updates_op)

        for grad, var in grads:
            tf.histogram_summary(var.op.name, var)
            try:
                tf.histogram_summary(var.op.name + '/gradients', grad)
            except:
                print var.op.name

        fetches.update({
            'reg_loss': reg_loss, 
            'train_op': train_op,
            'global_step': global_step
        })

    else:

        print_graph_ops()

    return fetches


def spatial_tranform(images1, images0, batch_size, subset, loc_net, xform_reg):
    
    with tf.name_scope(None): 
        images1 = tf.identity(images1, name='input_stn')
    
    if loc_net == 'simple':
        print 'using simple localization network'
        theta = simple_loc_net(images1, batch_size)
    
    if loc_net == 'inception':
        print 'using inception localization network'
        theta, _ = inception_model.inference(
            images = (images1-128)/128.,
            num_classes = 3,
            for_training = (subset == 'train'),
            restore_logits = (subset != 'train')
        )
        theta = tf.nn.tanh(theta)
    
    with tf.name_scope(None): 
        theta = tf.identity(theta, name='theta')     
    tf.histogram_summary('theta', theta)
    
    if subset == 'train':
        with tf.name_scope(None):
            theta_loss = tf.nn.l2_loss(theta, name='theta_loss')
        tf.scalar_summary('theta_loss', theta_loss)
        tf.add_to_collection('regularization_losses', xform_reg*theta_loss)

    theta = [tf.squeeze(i) for i in tf.split(0, batch_size, theta)]
    with tf.control_dependencies(images0 + theta):
        images2 = tf.pack([interp(i, j) for i, j in zip(images0, theta)])
    
    blackline = tf.zeros([batch_size, 299, 1, 3])
    whiteline = 255*tf.ones([batch_size, 299, 1, 3])
    images12 = tf.concat(2, [images1, whiteline, blackline, whiteline, images2])
    tf.image_summary('xform_pairs', images12, max_images=batch_size)
    
    return images2


def simple_loc_net(images, batch_size):

    images -= 128
    images /= 128
    
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,3,64], 
            dtype=tf.float32, stddev=1e-3), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1,2,2,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), 
            trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

    pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                         padding='SAME', name='pool1')
        
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,64], 
            dtype=tf.float32, stddev=1e-3), name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1,2,2,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), 
            trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
    
    pool2 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1],
                         padding='SAME', name='pool2')
    
    dim = np.prod(pool2.get_shape().as_list()[1:])
    
    flat = tf.reshape(pool2, [batch_size, dim])
    
    with tf.name_scope('fc1') as scope:
        weights = tf.Variable(tf.truncated_normal([dim, 3], dtype=tf.float32,
            stddev=1e-3), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[3], dtype=tf.float32), 
            name='biases')
        theta = tf.add(tf.matmul(flat, weights), biases, name=scope)
        theta = tf.nn.tanh(theta)
    
    return theta


def interp(u, theta):

    u = tf.cast(u, 'float32')

    h, w = tf.shape(u)[0], tf.shape(u)[1]
    paddings = tf.pack([
        tf.pack([tf.maximum(h,w)-h/2]*2), 
        tf.pack([tf.maximum(h,w)-w/2]*2),
        tf.constant([0,0])
    ])
    u = tf.pad(u, paddings)
    s, c0, c1 = tf.split(0, 3, theta)

    h, w = tf.cast(h, 'float32'), tf.cast(w, 'float32')
    IJ = tf.maximum(h,w)*((s+1)/4*np.linspace(-1,1,299) + 1)
    I, J = IJ+c0*h/2, IJ+c1*w/2

    len_kern = IJ[1]-IJ[0]

    I0 = tf.floor(I) - len_kern/2
    J0 = tf.floor(J) - len_kern/2

    dI = 1./len_kern*tf.expand_dims(I-I0, -1)
    dJ = 1./len_kern*tf.expand_dims(J-J0, 0)

    I0 = tf.cast(I0, 'int32')
    J0 = tf.cast(J0, 'int32')
    len_kern = tf.cast(len_kern, 'int32')
    I1 = I0 + len_kern
    J1 = J0 + len_kern

    u0  = tf.transpose(tf.gather(u,  I0), [1,0,2])
    u1  = tf.transpose(tf.gather(u,  I1), [1,0,2])
    u00 = tf.transpose(tf.gather(u0, J0), [1,0,2])
    u01 = tf.transpose(tf.gather(u0, J1), [1,0,2])
    u10 = tf.transpose(tf.gather(u1, J0), [1,0,2])
    u11 = tf.transpose(tf.gather(u1, J1), [1,0,2])

    v = u00
    v += (u10-u00) * tf.expand_dims(dI, -1)
    v += (u01-u00) * tf.expand_dims(dJ, -1)
    v += (u00+u11-u01-u10) * tf.expand_dims(tf.matmul(dI, dJ), -1)

    v.set_shape([299, 299, 3])
    
    return v