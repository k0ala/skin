"""A module for handling the skin dataset

Summary of available functions:

    - Repeat data samples to get even distribution over classes
    - Add images from auxillary class to dataset
    - Discard fraction of data from each class
    - Add mice images to pigmented malignant class
    - Load the skin dataset

"""

import numpy as np
from numpy import random as rnd
import json
import os
import time


def _apply_data_fraction(XY, p):
    """Discard fraction of data from each class

    Args:
        XY (numpy.array): 2D array of data
        p (float): Fraction of data to keep

    Returns:
        XY (numpy.array): 2D array of data
    """

    Y = XY[:,1]
    Ys = sorted(set(Y))
    C = np.array([sum(Y==y) for y in Ys])
    rnd.shuffle(XY)
    XY = np.concatenate([
        XY[Y==y][:int(c*p)] 
        for y, c in zip(Ys, C)
    ])

    return XY


def _evenly_distribute(XY):
    """Repeat data samples to get even distribution over classes

    Args:
        XY (numpy.array): 2D array of data

    Returns:
        XY (numpy.array): 2D array of data
    """

    Y = XY[:,1]
    Ys = sorted(set(Y))
    C = np.array([sum(Y==y) for y in Ys])
    XY = np.concatenate([
        np.tile(XY[Y==y], [max(C)/c+1, 1])[:max(C)]
        for y, c in zip(Ys, C)
    ])

    return XY


def _add_aux_class(XY, aux_class_path):
    """Add images from auxillary class to dataset

    Args:
        XY (numpy.array): 2D array of data
        aux_class_path (string): Path to auxillary images

    Returns:
        XY (numpy.array): 2D array of data
    """

    aux_images = [
        os.path.join(aux_class_path, i) 
        for i in os.listdir(aux_class_path)
    ]
    aux_class = os.path.normpath(aux_class_path).split('/')[-1]

    XY_aux = np.array([aux_images, [aux_class]*len(aux_images)]).T
    XY = np.concatenate([XY, XY_aux])

    return XY


def _add_mice(XY, p, meta, m2x):
    """Add mice images to pigmented malignant class

    Args:
        XY (numpy.array): 2D array of data
        p (float): Fraction of pigmented malignant images that will be mice
        meta (list): Mice meta entries
        m2x (function): Function to extract image path from meta entry

    Returns:
        XY (numpy.array): 2D array of data
    """

    Y = XY[:,1]
    y = 'pigmented malignant'
    M = [m for m in meta if m['database'] == 'mice']
    XY_mice = np.array([(m2x(m), y) for m in M])
    num_mice = int(p/(1-p)*sum(Y==y))
    XY = np.concatenate([XY, XY_mice[:num_mice]])

    return XY


def load_data(data_path, phase = 'train',
    tax_score = 0.8, skin_prob = 0.4, evenly_distribute = False, 
    data_fraction = None, aux_class_paths = None, mice_fraction = None):

    """Load the skin dataset

    Args:
        data_path (string): Path to ``meta.json`` and ``images`` directory.
        phase (string): Load either 'train' or 'test' dataset
        tax_score (float): Minimum allowed string matching score [0,1] between
            an image's title and the nearest disease in the taxonomy.  
        skin_prob (float): Minimum allowed skin probability [0,1] of an image.
        evenly_distribute (bool): Whether to duplicate examples until there is
            an even distribution among classes.
        data_fraction (float): Fraction [0,1] per class of data to use
        mice_fraction (float): Fraction [0,1] of pigmented malignant lesions
            that will be mice.
        aux_class_paths (list): A list of paths to images of of auxillary 
            classes to train on (e.g. healthy_skin, not_skin, etc).

    Returns:
        data (list): List of image path and label tuples. For example::

            [('images/img1.jpg', 'pigmented malignant'),
             ('images/img2.jpg', 'epidermal benign'),
             ('images/img2.jpg', 'inflammatory')]
    """

    meta = json.load(open(os.path.join(data_path, 'meta_{}.json'.format(phase))))

    M = [
        m for m in meta
        if 'tax_path' in m and m['tax_path_score'] > tax_score
        and 'skin_prob' in m and m['skin_prob'] > skin_prob
        and m['database'] != 'mice'
    ]

    labels = sorted({m['label'] for m in meta})
    num_classes = len(labels)
    
    m2x = lambda m: os.path.join(data_path, 'images', m['filename'])
    m2y = lambda m: m['label']

    XY = np.array([(m2x(m), m2y(m)) for m in M])

    if aux_class_paths:
        for aux_class_path in aux_class_paths:
            XY = _add_aux_class(XY, aux_class_path)

    if data_fraction: 
        XY = _apply_data_fraction(XY, data_fraction)

    if mice_fraction:
        XY = _add_mice(XY, mice_fraction, meta, m2x)
    
    if evenly_distribute:
        XY = _evenly_distribute(XY)

    rnd.shuffle(XY)
    data = [(str(x), str(y)) for x, y in XY]

    return data