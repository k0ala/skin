"""Script to start training procedure
"""

import os
import json
import tensorflow as tf
import skin


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name = 'net_path',
    default_value = None,
    docstring = 'Directory to save stuff'
)
tf.app.flags.DEFINE_integer(
    flag_name = 'batch_size',
    default_value = 100,
    docstring = 'Number of images in a batch'
)
tf.app.flags.DEFINE_float(
    flag_name = 'learning_rate',
    default_value = 1e-2,
    docstring = 'Learning rate'
)
tf.app.flags.DEFINE_integer(
    flag_name = 'gpu',
    default_value = 0,
    docstring = 'GPU to use'
)
tf.app.flags.DEFINE_float(
    flag_name = 'gpu_fraction',
    default_value = 0.95,
    docstring = 'fraction of GPU to use'
)
tf.app.flags.DEFINE_string(
    flag_name = 'split',
    default_value = 'trainall',
    docstring = 'Which train/val split to use'
)
tf.app.flags.DEFINE_boolean(
    flag_name = 'spatial_transformer',
    default_value = False,
    docstring = 'Whether to train a spatial tranform'
)
tf.app.flags.DEFINE_string(
    flag_name = 'loc_net',
    default_value = 'fc',
    docstring = 'Type of localization net (fc, conv, inception)'
)
tf.app.flags.DEFINE_float(
    flag_name = 'loc_net_reg',
    default_value = 1,
    docstring = 'Regularization for localization net weights'
)
tf.app.flags.DEFINE_float(
    flag_name = 'xform_reg',
    default_value = 0.01,
    docstring = 'Regularization for image transform'
)


def main():

    net_hypes = {
        'batch_size': FLAGS.batch_size,
        'variable_averages_decay': 0.99,
        'learning_rate': FLAGS.learning_rate,
        'epsilon': 0.1,
        'beta1': 0.9,
        'beta2': 0.99,
        'spatial_transformer': FLAGS.spatial_transformer,
        'loc_net': FLAGS.loc_net,
        'loc_net_reg': FLAGS.loc_net_reg,
        'xform_reg': FLAGS.xform_reg
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    dataset = skin.dataset.DATA_PATH + '/datasets/' + FLAGS.split
    dataset = skin.DataSet(dataset, subset='train')
    labels = sorted({y for x, y in dataset.train})
    net = skin.ConvNet(FLAGS.net_path, net_hypes, labels)
    net.train(dataset.train, gpu=0, gpu_fraction=FLAGS.gpu_fraction)

if __name__ == '__main__': main()
