"""Script to run validation
"""

import os
import json
import tensorflow as tf
import skin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    flag_name = 'net_path', 
    default_value = None, 
    docstring = 'Directory of net to evaluate'
)
tf.app.flags.DEFINE_integer(
    flag_name = 'gpu', 
    default_value = 0, 
    docstring = 'GPU to use'
)
tf.app.flags.DEFINE_float(
    flag_name = 'gpu_fraction', 
    default_value = 0.8, 
    docstring = 'fraction of GPU to use'
)

def main():

    net = skin.ConvNet(FLAGS.net_path)
    dataset = skin.dataset.DATA_PATH + '/datasets/' + net.hypes['split']
    dataset = skin.DataSet(dataset, subset='val')
    net.validate(dataset.val, gpu=FLAGS.gpu, gpu_fraction=FLAGS.gpu_fraction)

if __name__ == '__main__': main()
