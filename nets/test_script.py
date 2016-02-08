"""Script to start evaluation procedure
"""

import os
import json
import tensorflow as tf
import skin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name = 'net_name', 
    default_value = '/tmp/brettnet', 
    docstring = 'Directory of net to evaluate'
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

    data = skin.data.load_data(
        data_path = '../data/',
        phase = 'test',
        tax_score = 0,
        skin_prob = 0,
        evenly_distribute = False
    )

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    with open(net_name + '/data_test.txt', 'w') as f: 
        json.dump(data, f, indent=4)

    skin.test.test(
        net_name = net_name,
        data = data,
        gpu_fraction = FLAGS.gpu_fraction
    )

if __name__ == '__main__': main()
