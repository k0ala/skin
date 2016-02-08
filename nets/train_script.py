"""Script to start training procedure
"""

import os
import json
import skin
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name = 'net_name', 
    default_value = '/tmp/brettnet', 
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
    default_value = True, 
    docstring = 'Whether to use dropout on final layer features'
)
tf.app.flags.DEFINE_integer(
    flag_name = 'gpu', 
    default_value = 0, 
    docstring = 'GPU to use'
)
tf.app.flags.DEFINE_float(
    flag_name = 'gpu_fraction', 
    default_value = 0.5, 
    docstring = 'Fraction of GPU memory to use'
)

def main():

    net_name = FLAGS.net_name

    hypes = {
        'base_net': FLAGS.base_net,
        'batch_size': FLAGS.batch_size,
        'optimizer': 'RMSProp',
        'learning_rate': FLAGS.learning_rate,
        'epsilon': 0.1,
        'weight_decay': FLAGS.weight_decay,
        'use_dropout': FLAGS.dropout,
        'loss_matrix': None,
        'data_fraction': 1,
        'mice_fraction': 0.05,
        'min_skin_prob': 0.4,
        'min_tax_score': 0.8,
        'evenly_distribute': True
    }

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
        aux_class_paths = ['../data/notskin']
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    if not os.path.exists(net_name): os.makedirs(net_name)
    with open(net_name + '/hypes.txt', 'w') as f: json.dump(hypes, f, indent=4)
    with open(net_name + '/data_train.txt', 'w') as f: 
        json.dump(data, f, indent=4)
    with open(net_name + '/label_names.txt', 'w') as f:
        json.dump(sorted({y for x, y in data}), f, indent=4)

    skin.train.train(
        net_name = net_name,
        data = data,
        gpu_fraction = FLAGS.gpu_fraction
    )

if __name__ == '__main__': main()
