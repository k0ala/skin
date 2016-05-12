"""
Utility functions
"""

import os
import json
import numpy as np
import tensorflow as tf
from scipy import stats
import sklearn.metrics
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot, rcParams, patches
from inception import inception_model
from spatial_transformer import spatial_transformer
import skin

rcParams.update({'figure.autolayout': True})


def spatial_tranform(images0, batch_size, subset, loc_net, xform_reg):

    images1 = tf.pack([
        tf.image.resize_images(i, 299, 299)
        for i in images0
    ])

    with tf.name_scope(None):
        images1 = tf.identity(images1, name='input_stn')

    with tf.variable_scope('loc_net') as scope:

        if loc_net == 'fc':
            print 'using fully connected localization network'
            theta = loc_net_fc(images1, batch_size)

        if loc_net == 'conv':
            print 'using convolutional localization network'
            theta = loc_net_conv(images1, batch_size)

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
    tf.histogram_summary('theta/zoom', theta[:,0])
    tf.histogram_summary('theta/pan_horizontal', theta[:,1])
    tf.histogram_summary('theta/pan_vertical', theta[:,2])

    if subset == 'train':
        with tf.name_scope(None):
            theta_loss = tf.nn.l2_loss(theta, name='theta_loss')
        tf.scalar_summary('theta_loss', theta_loss)
        tf.add_to_collection('regularization_losses', xform_reg*theta_loss)

    images2 = []
    for i in range(batch_size):
        s, dx, dy = (theta[i,0]+1)/2, theta[i,1], theta[i,2]
        th = tf.pack([s, 0, dx,
                      0, s, dy])
        u = images0[i]
        u, th = tf.expand_dims(u, 0), tf.expand_dims(th, 0)
        dsf = tf.cast(tf.shape(u)[1], 'float32') / 299
        v = spatial_transformer.transformer(u, th, dsf)
        v = tf.image.resize_images(v[0,:,:,:], 299, 299)
        v.set_shape([299, 299, 3])
        images2.append(v)
    images2 = tf.pack(images2)

    images12 = tf.concat(2, [images1, images2])
    blkbar = tf.zeros([batch_size, 299/2, 299*2, 3])
    whtbar = 255*tf.ones([batch_size, 299/2, 299*2, 3])
    images12 = tf.concat(1, [whtbar, images12, whtbar])
    images12 = tf.clip_by_value(images12, 0, 255)
    tf.image_summary('xform_pairs', images12, max_images=batch_size)

    return images2


def loc_net_fc(images, batch_size):

    images -= 128
    images /= 128.

    images = tf.image.resize_images(images, 150, 150)
    images_flat = tf.reshape(images, [batch_size, -1])
    hidden_size = 100

    with tf.name_scope('fc1') as scope:
        weights = tf.Variable(tf.truncated_normal([150**2*3, hidden_size],
            dtype=tf.float32, stddev=1e-3), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[hidden_size],
            dtype=tf.float32), name='biases')
        hidden = tf.add(tf.matmul(images_flat, weights), biases, name=scope)
        hidden = tf.nn.relu(hidden)

    with tf.name_scope('fc2') as scope:
        weights = tf.Variable(tf.truncated_normal([hidden_size, 3],
            dtype=tf.float32, stddev=1e-3), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[3], dtype=tf.float32),
            name='biases')
        theta = tf.add(tf.matmul(hidden, weights), biases, name=scope)
        theta = tf.nn.tanh(theta)

    return theta


def loc_net_conv(images, batch_size):

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


def square_pad(x):
    """Pads image with black into the smallest circumscribing square"""
    h, w = tf.shape(x)[0], tf.shape(x)[1]
    d = tf.maximum(h, w)
    paddings = tf.pack([
        tf.pack([(d-h)/2]*2),
        tf.pack([(d-w)/2]*2),
        tf.constant([0,0])
    ])
    x = tf.pad(x, paddings)
    x.set_shape([None, None, 3])
    return x

def init_logits(path, filenames, labels, subset):

    logits_info = {i: j for i, j in zip(filenames, labels)}
    logits_info = {
        'filenames': logits_info.keys(),
        'labels': logits_info.values()
    }

    logits_matrix = np.zeros(
        shape = [len(logits_info['filenames']), len(set(labels))],
        dtype = 'float16'
    )
    logits_matrix[:] = np.nan

    with open(path + '/logits_info_{}.txt'.format(subset), 'w') as f:
        json.dump(logits_info, f, indent=4)

    return logits_matrix, logits_info


def save_matrix(path, A, name, subset, step):

    path = os.path.join(path, name, subset)
    if not os.path.exists(path): os.makedirs(path)
    np.save(path+'/step{}.npy'.format(step), A)


def get_hist(p, y):
    """Create histogram of benign and malignant probabilities

    Args:
        p (numpy.array): 1D array of malignant probabilities
        y (numpy.array): 1D array of label integers

    Returns:
        fig (matplotlib.figure): The histogram
    """

    nbins = 30
    bins = np.linspace(0, 1, nbins)

    fig = pyplot.figure(figsize=(8,8))
    pyplot.hist(p[y==0], bins=bins, alpha=0.5, normed=True, facecolor='green')
    pyplot.hist(p[y==1], bins=bins, alpha=0.5, normed=True, facecolor='red')
    pyplot.xlim(0, 1)
    pyplot.ylim(0, nbins/2)
    pyplot.ylabel('normalized count', fontsize=18, labelpad=20)
    pyplot.xlabel('probability malignant', fontsize=18, labelpad=10)

    return fig


def get_roc(p, y, data=None, confidence_interval=None):
    """Create sensitiviy/specificity curve

    Args:
        p (numpy.array): 1D array of malignant probabilities
        y (numpy.array): 1D array of label integers
        data (dict): Other data to plot (e.g. dermatologists results)
        confidence_interval (float): Confidence interval [0,1] of
            uncertainty ellipses

    Returns:
        fig (matplotlib.figure): The SS curve
    """

    auc = sklearn.metrics.roc_auc_score(y, p)

    thresholds = np.sort(np.concatenate([
        np.linspace(0, 1, 50),
        np.logspace(0, -5, 50),
        1 - np.logspace(0, -5, 50)
    ]))

    plot_kwargs = {
        'net': {'c':'b', 'ls': '-', 'marker': 'o'},
        'derms': {'c':'r', 'ls': 'None', 'marker': 'o'},
        'novices': {'c':'y', 'ls': 'None', 'marker': 'o'},
        'melafind': {'c': '0.5', 'ls': '-', 'marker': 'o'}
    }

    if data is None: data = {}

    data['net'] = [
        {
            'specificity': np.mean(p[y==0]<t),
            'sensitivity': np.mean(p[y==1]>t),
            'num_benign': sum(y==0),
            'num_malignant': sum(y==1)
        }
        for t in thresholds
    ]

    fig = pyplot.figure(figsize=(8,8))
    ax = pyplot.subplot(111, aspect='equal')

    for group in data:

        p0 = [i['specificity'] for i in data[group]]
        p1 = [i['sensitivity'] for i in data[group]]

        pyplot.plot(p0, p1, **plot_kwargs[group])

        if confidence_interval:

            n0 = [i['num_benign'] for i in data[group]]
            n1 = [i['num_malignant'] for i in data[group]]

            p0_intervals = [
                stats.beta.interval(confidence_interval, p*n+1, (1-p)*n+1)
                for p, n in zip(p0, n0) if n > 0
            ]

            p1_intervals = [
                stats.beta.interval(confidence_interval, p*n+1, (1-p)*n+1)
                for p, n in zip(p1, n1) if n > 0
            ]

            for dp0, dp1 in zip(p0_intervals, p1_intervals):

                ax.add_artist(patches.Ellipse(
                    xy = [
                        np.mean(dp0),
                        np.mean(dp1)
                    ],
                    width = np.diff(dp0),
                    height = np.diff(dp1),
                    facecolor = 'none',
                    linestyle = 'dotted',
                    edgecolor = plot_kwargs[group]['c']
                ))

    pyplot.ylabel('sensitivity', fontsize=18, labelpad=30)
    pyplot.xlabel('specificity', fontsize=18, labelpad=30)
    pyplot.title('%.2f AUC' % auc, fontsize=22)

    return fig

def make_plots(logits, labels, labelset, figpath, conf_inter=0.8):
    """Make plots and save to disk
    """

    if not os.path.exists(figpath): os.makedirs(figpath)

    def save_fig(fig, category, figtype):
        fig.savefig(fig_path + '/{}_{}.svg'.format(figtype, category))

    with open(skin.dataset.DATA_PATH+'/competition.txt') as f:
        competition_data = json.load(f)

    b = np.logical_not(np.isnan(logits.sum(1)))
    Y, Z = np.array(labels)[b], logits[b].astype('float32')

    for category in ['pigmented', 'epidermal']:

        i = labelset.index('{} benign'.format(category))
        j = labelset.index('{} malignant'.format(category))

        if category not in competition_data:
            competition_data[category] = None

        IJ = np.logical_or(Y==i, Y==j)
        y = Y[IJ].copy()
        y[y==i] = 0
        y[y==j] = 1

        p = np.exp(Z[IJ][:,[i,j]])
        p = p[:,1]/p.sum(axis=1)

        histfig = skin.util.get_hist(p, y)
        histfig.savefig(figpath + '/{}_{}.svg'.format('hist', category))

        rocfig = skin.util.get_roc(
            p, y, competition_data[category], conf_inter)
        rocfig.savefig(figpath + '/{}_{}.svg'.format('roc', category))

        pyplot.close('all')
