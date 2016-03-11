"""Script to start training procedure
"""

import os
import json
import tensorflow as tf
import skin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name = 'net_name', 
    default_value = None, 
    docstring = 'Directory to save stuff'
)
tf.app.flags.DEFINE_string(
    flag_name = 'base_net', 
    default_value = 'inception_v1', 
    docstring = 'Net to finetune, either inception_v1 or inception_v3'
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
tf.app.flags.DEFINE_float(
    flag_name = 'weight_decay', 
    default_value = 1e-4, 
    docstring = 'Weight decay'
)
tf.app.flags.DEFINE_boolean(
    flag_name = 'dropout', 
    default_value = False, 
    docstring = 'Whether to use dropout on final layer features'
)
tf.app.flags.DEFINE_integer(
    flag_name = 'gpu', 
    default_value = 0, 
    docstring = 'GPU to use'
)
tf.app.flags.DEFINE_string(
    flag_name = 'optimizer', 
    default_value = 'RMSProp', 
    docstring = 'RMSProp or Adam'
)
tf.app.flags.DEFINE_boolean(
    flag_name = 'batch_normalize', 
    default_value = False, 
    docstring = 'Whether to use batch normalization'
)
tf.app.flags.DEFINE_string(
    flag_name = 'split', 
    default_value = 'dt', 
    docstring = 'Which train/test split to use'
)


def main():

    hypes = {
        'base_net': FLAGS.base_net,
        'batch_size': FLAGS.batch_size,
        'batch_normalize': FLAGS.batch_normalize,
        'optimizer': FLAGS.optimizer,
        'learning_rate': FLAGS.learning_rate,
        'epsilon': 1.0,
        'weight_decay': FLAGS.weight_decay,
        'use_dropout': FLAGS.dropout,
        'loss_matrix': None,
        'data_fraction': 1,
        'mice_fraction': 0.1,
        'min_skin_prob': 0.4,
        'min_tax_score': 0.85,
        'evenly_distribute': True,
        'split': FLAGS.split
    }

    if hypes['base_net'] == 'inception_v3':
        hypes['batch_normalize'] = True

    if hypes['base_net'] == 'inception_v1': 
        hypes['head_weights'] = [0.2, 0.3, 0.5]

    data = skin.data.load_data(
        data_path = '../data/',
        phase = 'train',
        tax_score = hypes['min_tax_score'],
        skin_prob = hypes['min_skin_prob'],
        evenly_distribute = hypes['evenly_distribute'],
        data_fraction = hypes['data_fraction'],
        mice_fraction = hypes['mice_fraction'],
        aux_class_paths = ['../data/notskin'],
        split = hypes['split']
    )

    net = skin.ConvNet(
        name = FLAGS.net_name,
        labels = sorted({y for x, y in data}),
        hypes = hypes
    )

    with open(net.name + '/data_train.txt', 'w') as f: 
        json.dump(data, f, indent=4)

    net.train(data, gpu=FLAGS.gpu, gpu_fraction=0.8)

if __name__ == '__main__': main()
