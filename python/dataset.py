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

DATA_PATH = '/home/kuprel/skin/data'


class DataSet(object):


    def __init__(self, path, params=None, subset=None):
        self.path = path
        if os.path.exists(self.path):
            self.restore(subset)
        else:
            os.makedirs(self.path)
            self.params = params
            self.create()
            self.save()


    def save(self):

        with open(self.path + '/params.txt', 'w') as f:
            json.dump(self.params, f, indent=4)
        with open(self.path + '/train.txt', 'w') as f:
            json.dump(self.train, f, indent=4)
        with open(self.path + '/val.txt', 'w') as f:
            json.dump(self.val, f, indent=4)


    def restore(self, subset):

        print 'restoring dataset', self.path

        with open(self.path+'/params.txt') as f: self.params = json.load(f)

        if subset == 'train':
            with open(self.path+'/train.txt') as f: self.train = json.load(f)
        elif subset == 'val':
            with open(self.path+'/val.txt') as f: self.val = json.load(f)
        else:
            with open(self.path+'/train.txt') as f: self.train = json.load(f)
            with open(self.path+'/val.txt') as f: self.val = json.load(f)


    def create(self):
        print 'creating', self.path
        self.train = self.load('train')
        self.val = self.load('val')


    def load(self, subset):

        meta = '/meta_{}_{}.json'.format(self.params['split'], subset)
        meta = json.load(open(DATA_PATH + meta))

        M = [
            m for m in meta
            if 'tax_path' in m and m['tax_path_score'] > self.params['min_tax_score']
            and 'skin_prob' in m and m['skin_prob'] > self.params['min_skin_prob']
            and m['database'] != 'mice'
        ]

        labels = sorted({m['label'] for m in meta})
        num_classes = len(labels)

        m2x = lambda m: DATA_PATH+'/images/'+m['filename']
        m2y = lambda m: m['label']

        XY = np.array([(m2x(m), m2y(m)) for m in M])

        if subset == 'train':

            if self.params['aux_class_paths']:
                XY = self.add_aux_classes(XY)

            if self.params['data_fraction'] != 1:
                XY = self.apply_data_fraction(XY)

            if self.params['mice_fraction']:
                XY = self.add_mice(XY, meta)

            if self.params['evenly_distribute']:
                XY = self.evenly_distribute(XY)

        rnd.shuffle(XY)

        data = [(str(x), str(y)) for x, y in XY]

        return data


    def apply_data_fraction(self, XY):
        """Discard fraction of data from each class

        Args:
            XY (numpy.array): 2D array of data
            p (float): Fraction of data to keep

        Returns:
            XY (numpy.array): 2D array of data
        """

        print 'applying data fraction'

        p = self.params['data_fraction']

        Y = XY[:,1]
        Ys = sorted(set(Y))
        C = np.array([sum(Y==y) for y in Ys])
        rnd.shuffle(XY)
        XY = np.concatenate([
            XY[Y==y][:int(c*p)]
            for y, c in zip(Ys, C)
        ])

        return XY


    def evenly_distribute(self, XY):
        """Repeat data samples to get even distribution over classes

        Args:
            XY (numpy.array): 2D array of data

        Returns:
            XY (numpy.array): 2D array of data
        """

        print 'evenly distributing'

        Y = XY[:,1]
        Ys = sorted(set(Y))
        C = np.array([sum(Y==y) for y in Ys])
        XY = np.concatenate([
            np.tile(XY[Y==y], [max(C)/c+1, 1])[:max(C)]
            for y, c in zip(Ys, C)
        ])

        return XY


    def add_aux_classes(self, XY):
        """Add images from auxillary class to dataset

        Args:
            XY (numpy.array): 2D array of data
            aux_class_path (string): Path to auxillary images

        Returns:
            XY (numpy.array): 2D array of data
        """

        print 'adding auxillary class'

        for aux_class_path in self.params['aux_class_paths']:

            aux_images = [
                os.path.join(aux_class_path, i)
                for i in os.listdir(aux_class_path)
            ]
            aux_class = os.path.normpath(aux_class_path).split('/')[-1]

            XY_aux = np.array([aux_images, [aux_class]*len(aux_images)]).T
            XY = np.concatenate([XY, XY_aux])

        return XY


    def add_mice(self, XY, meta):
        """Add mice images to pigmented malignant class

        Args:
            XY (numpy.array): 2D array of data
            p (float): Fraction of pigmented malignant images that will be mice
            meta (list): Mice meta entries
            m2x (function): Function to extract image path from meta entry

        Returns:
            XY (numpy.array): 2D array of data
        """

        print 'adding mice'

        m2x = lambda m: DATA_PATH+'/images/'+m['filename']
        p = self.params['mice_fraction']

        Y = XY[:,1]
        y = 'pigmented malignant'
        M = [m for m in meta if m['database'] == 'mice']
        XY_mice = np.array([(m2x(m), y) for m in M])
        num_mice = int(p/(1-p)*sum(Y==y))
        XY = np.concatenate([XY, XY_mice[:num_mice]])

        return XY
