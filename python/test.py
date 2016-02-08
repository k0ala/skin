"""Tests network
"""

import tensorflow as tf
import numpy as np
import os
import time
import json
import sys
import skin


def _get_step(net_name):
    """Get iteration of most recent checkpoint that has not been evaluated yet

    Args:
        net_name (string): Directory where net stuff is saved

    Returns:
        step (int): Next iteration to evaluate if one exists
    """
    net_files = os.listdir(net_name)
    s = lambda i: int(i.split('-')[-1])
    steps = sorted([
        s(i) for i in net_files
        if 'save.ckpt' in i
        and '.meta' not in i
        and 'logits_test_{}.npy'.format(s(i)) not in net_files
    ])

    if steps:
        return steps[-1]
    else:
        return None


def test(net_name, data, gpu_fraction=0.5):
    """Train net

    Args:
        net_name (string): Directory where net stuff is saved
        data (list): Dataset as list of tuples, see ``data.load_data``
        gpu_fraction (float): Fraction of GPU memory to use
    """

    phase = 'test'

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
        num_threads = 3,
        distort = False
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
                    loss_matrix = hypes['loss_matrix']
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

    fetches = {
        'loss': loss,
        'filenames': filenames,
        'logits': logits
    }

    coord = tf.train.Coordinator()
    saver = tf.train.Saver()

    print_str = ', '.join([
        'Evaluated %d of %d images',
        'Train Step: %d',
        'Loss: %.2f',
        'Time/image (ms): %.1f'
    ])

    num_examples = len(data)

    with tf.Session(config=config) as sess:

        tf.initialize_all_variables().run()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        while True:

            step = _get_step(net_name)
            if step is None:
                print 'No checkpoints to evaluate, trying again in 1 minute'
                time.sleep(60)
                continue
            else:
                print 'Evaluating checkpoint from step {}'.format(step)


            saver.restore(sess, net_name+'/save.ckpt-'+str(step))

            logits_matrix[:] = np.nan
            num_evaluated = 0

            try:

                while (
                    num_evaluated < num_examples + 4*hypes['batch_size']
                    and not coord.should_stop()
                ):

                    t = time.time()

                    fetch_names = sorted(fetches.keys())
                    fetched = sess.run([fetches[i] for i in fetch_names])
                    fetched = dict(zip(fetch_names, fetched))

                    dt = time.time()-t

                    I = [
                        logits_info['filenames'].index(i) 
                        for i in fetched['filenames']
                    ]
                    logits_matrix[I] = fetched['logits']

                    print print_str % (
                        num_evaluated,
                        num_examples,
                        step,
                        fetched['loss'],
                        dt/hypes['batch_size']*1000
                    )

                    num_evaluated += hypes['batch_size']

                np.save(
                    net_name+'/logits_{}_{}.npy'.format(phase, step), 
                    logits_matrix
                )
                print sum(np.isnan(logits_matrix[:,0])), 'missed images'

                skin.analysis.make_plots(net_name, step, phase)

            except Exception as e:
                coord.request_stop(e)

    coord.request_stop()
    coord.join(threads)