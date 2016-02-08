"""Trains network
"""

import tensorflow as tf
import numpy as np
import os
import time
import json
import skin


def _restore_net(net_name):
    """Restore checkpoint and stored logits

    Args:
        net_name (string): Directory where net stuff is saved

    Returns:
        ckpt, logits (tuple): Last checkpoint and logits
    """

    ckpt = tf.train.get_checkpoint_state(net_name)
    logits = None

    if ckpt:

        ckpt_file = ckpt.model_checkpoint_path
        print('found checkpoint: {}'.format(ckpt_file))
        step = ckpt_file.split('/')[-1].split('-')[-1]
        logits_file = net_name+'/logits_train_{}.npy'.format(step)
        if os.path.exists(logits_file):
            print ('found logits')
            logits = np.load(logits_file)

    return ckpt, logits


def train(net_name, data, gpu_fraction=0.5):
    """Train net

    Args:
        net_name (string): Directory where net stuff is saved
        data (list): Dataset as list of tuples, see ``data.load_data``
        gpu_fraction (float): Fraction of GPU memory to use
    """

    ckpt, logits_matrix = _restore_net(net_name)

    phase = 'train'

    with open(net_name + '/hypes.txt') as f:
        hypes = json.load(f)
    with open(net_name + '/label_names.txt') as f:
        label_names = json.load(f)

    filenames = [x for x, y in data]
    labels = [label_names.index(y) for x, y in data]

    logits_info = {filename: label for filename, label in zip(filenames, labels)}
    logits_info = {
        'filenames': logits_info.keys(),
        'labels': logits_info.values()
    }

    with open(net_name + '/logits_info_{}.txt'.format(phase), 'w') as f:
        json.dump(logits_info, f, indent=4)

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction = gpu_fraction
        )
    )

    if logits_matrix is None:
        logits_matrix = np.zeros(
            [len(logits_info['filenames']), len(label_names)], 
            dtype='float16'
        )
        logits_matrix[:] = np.nan

    images, labels, filenames = skin.inference.get_inputs(
        filenames = filenames,
        labels = labels, 
        batch_size = hypes['batch_size'],
        base_net = hypes['base_net'],
        num_threads = 6,
        distort = True
    )

    features = skin.inference.get_features(
        images = images,
        base_net = hypes['base_net'],
        sess_config = config
    )

    if type(features) is list:

        loss = []

        for i, features in enumerate(features):

            if hypes['use_dropout']:
                features = tf.nn.dropout(features, 0.5)

            with tf.name_scope('logits_{}'.format(i)):
                logits = skin.inference.get_logits(
                    features = features,
                    num_classes = len(label_names)
                )

            with tf.name_scope('loss_{}'.format(i)):
                loss.append(skin.inference.get_loss(
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

        if hypes['use_dropout']:
            features = tf.nn.dropout(features, 0.5)

        with tf.name_scope('logits'):
            logits = skin.inference.get_logits(
                features = features,
                num_classes = len(label_names)
            )

        with tf.name_scope('loss'):
            loss = skin.inference.get_loss(
                logits = logits, 
                labels = labels,
                loss_matrix = hypes['loss_matrix']
            )

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

    opt = eval('tf.train.{}Optimizer'.format(hypes['optimizer']))(
        learning_rate = hypes['learning_rate'], 
        epsilon = hypes['epsilon']
    )

    global_step = tf.Variable(0, name='global_step', trainable=False)
    grads = opt.compute_gradients(regularized_loss)

    train_op = opt.apply_gradients(
        grads_and_vars = grads, 
        global_step = global_step
    )

    tf.histogram_summary(features.op.name, features)
    tf.histogram_summary(logits.op.name, logits)
    tf.scalar_summary(loss.op.name, loss)
    tf.scalar_summary(weights_norm.op.name, weights_norm)
    for grad, var in grads:
        tf.histogram_summary(var.op.name, var)
        tf.histogram_summary(var.op.name + '/gradients', grad)
    summary_op = tf.merge_all_summaries()

    fetches = {
        'loss': loss,
        'filenames': filenames,
        'logits': logits,
        'train_op': train_op,
        'weights_norm': weights_norm
    }

    coord = tf.train.Coordinator()
    savers = {
        'short_term': tf.train.Saver(max_to_keep=3, name='save_short'),
        'long_term':  tf.train.Saver(max_to_keep=100, name='save_long')
    }
    writer = tf.train.SummaryWriter(net_name)
    writer.add_graph(tf.get_default_graph().as_graph_def())

    print_str = ', '.join([
        'Step: %d',
        'Loss: %.2f',
        'Weights norm: %.2f',
        'Time/image (ms): %.1f'
    ])

    with tf.Session(config=config) as sess:

        tf.initialize_all_variables().run()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if ckpt: savers['short_term'].restore(sess, ckpt.model_checkpoint_path)

        try:

            while not coord.should_stop():

                t = time.time()

                fetch_names = sorted(fetches.keys())
                fetched = sess.run([fetches[i] for i in fetch_names])
                fetched = dict(zip(fetch_names, fetched))

                dt = time.time()-t

                step = global_step.eval()

                I = [
                    logits_info['filenames'].index(i) 
                    for i in fetched['filenames']
                ]
                logits_matrix[I] = fetched['logits']

                print print_str % (
                    step,
                    fetched['loss'],
                    fetched['weights_norm'],
                    dt/hypes['batch_size']*1000
                )

                if step%100 == 0:

                    print('saving')

                    savers['short_term'].save(
                        sess = sess, 
                        save_path = os.path.join(net_name, 'save.ckpt'),
                        global_step = global_step
                    )

                    writer.add_summary(sess.run(summary_op), global_step=step)

                    np.save(
                        net_name+'/logits_{}_{}.npy'.format(phase, step), 
                        logits_matrix
                    )

                    skin.analysis.make_plots(net_name, step, phase)

                if step%1000 == 0:

                    savers['long_term'].save(
                        sess = sess, 
                        save_path = os.path.join(net_name, 'SAVE.ckpt'),
                        global_step = global_step
                    )

        except Exception as e: coord.request_stop(e)

    coord.join(threads)