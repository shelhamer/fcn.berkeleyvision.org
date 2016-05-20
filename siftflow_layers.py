import caffe

import numpy as np
from PIL import Image
import scipy.io

import random

class SIFTFlowSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from SIFT Flow
    one-at-a-time while reshaping the net to preserve dimensions.

    This data layer has three tops:

    1. the data, pre-processed
    2. the semantic labels 0-32 and void 255
    3. the geometric labels 0-2 and void 255

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - siftflow_dir: path to SIFT Flow dir
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for semantic segmentation of object and geometric classes.

        example: params = dict(siftflow_dir="/path/to/siftflow", split="val")
        """
        # config
        params = eval(self.param_str)
        self.siftflow_dir = params['siftflow_dir']
        self.split = params['split']
        self.mean = np.array((114.578, 115.294, 108.353), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # three tops: data, semantic, geometric
        if len(top) != 3:
            raise Exception("Need to define three tops: data, semantic label, and geometric label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/{}.txt'.format(self.siftflow_dir, self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label_semantic = self.load_label(self.indices[self.idx], label_type='semantic')
        self.label_geometric = self.load_label(self.indices[self.idx], label_type='geometric')
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label_semantic.shape)
        top[2].reshape(1, *self.label_geometric.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label_semantic
        top[2].data[...] = self.label_geometric

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open('{}/Images/spatial_envelope_256x256_static_8outdoorcategories/{}.jpg'.format(self.siftflow_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx, label_type=None):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        if label_type == 'semantic':
            label = scipy.io.loadmat('{}/SemanticLabels/spatial_envelope_256x256_static_8outdoorcategories/{}.mat'.format(self.siftflow_dir, idx))['S']
        elif label_type == 'geometric':
            label = scipy.io.loadmat('{}/GeoLabels/spatial_envelope_256x256_static_8outdoorcategories/{}.mat'.format(self.siftflow_dir, idx))['S']
            label[label == -1] = 0
        else:
            raise Exception("Unknown label type: {}. Pick semantic or geometric.".format(label_type))
        label = label.astype(np.uint8)
        label -= 1  # rotate labels so classes start at 0, void is 255
        label = label[np.newaxis, ...]
        return label.copy()
