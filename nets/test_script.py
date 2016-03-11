"""Script to start evaluation procedure
"""

import os
import json
import tensorflow as tf
import skin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name = 'net_name', 
    default_value = None, 
    docstring = 'Directory of net to evaluate'
)
tf.app.flags.DEFINE_integer(
    flag_name = 'gpu', 
    default_value = 0, 
    docstring = 'GPU to use'
)

def main():

    net = skin.ConvNet(FLAGS.net_name)

    data = skin.data.load_data(
        data_path = '../data/',
        phase = 'test',
        tax_score = 0.85,
        skin_prob = 0.4,
        evenly_distribute = False,
        split = net.hypes['split']
    )

    with open(net.name + '/data_test.txt', 'w') as f: 
        json.dump(data, f, indent=4)

    net.test(data, gpu=FLAGS.gpu, gpu_fraction=0.8)

if __name__ == '__main__': main()
