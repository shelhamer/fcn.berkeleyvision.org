import caffe

import numpy as np
from PIL import Image
import scipy.io

import random

class PASCALContextSegDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from PASCAL-Context
    one-at-a-time while reshaping the net to preserve dimensions.

    The labels follow the 59 class task defined by

        R. Mottaghi, X. Chen, X. Liu, N.-G. Cho, S.-W. Lee, S. Fidler, R.
        Urtasun, and A. Yuille.  The Role of Context for Object Detection and
        Semantic Segmentation in the Wild.  CVPR 2014.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - voc_dir: path to PASCAL VOC dir (must contain 2010)
        - context_dir: path to PASCAL-Context annotations
        - split: train / val / test
        - randomize: load in random order (default: True)
        - seed: seed for randomization (default: None / current time)

        for PASCAL-Context semantic segmentation.

        example: params = dict(voc_dir="/path/to/PASCAL", split="val")
        """
        # config
        params = eval(self.param_str)
        self.voc_dir = params['voc_dir'] + '/VOC2010'
        self.context_dir = params['context_dir']
        self.split = params['split']
        self.mean = np.array((104.007, 116.669, 122.679), dtype=np.float32)
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)

        # load labels and resolve inconsistencies by mapping to full 400 labels
        self.labels_400 = [label.replace(' ','') for idx, label in np.genfromtxt(self.context_dir + '/labels.txt', delimiter=':', dtype=None)]
        self.labels_59 = [label.replace(' ','') for idx, label in np.genfromtxt(self.context_dir + '/59_labels.txt', delimiter=':', dtype=None)]
        for main_label, task_label in zip(('table', 'bedclothes', 'cloth'), ('diningtable', 'bedcloth', 'clothes')):
            self.labels_59[self.labels_59.index(task_label)] = main_label

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Main/{}.txt'.format(self.voc_dir,
                self.split)
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
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

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
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.voc_dir, idx))
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        The full 400 labels are translated to the 59 class task labels.
        """
        label_400 = scipy.io.loadmat('{}/trainval/{}.mat'.format(self.context_dir, idx))['LabelMap']
        label = np.zeros_like(label_400, dtype=np.uint8)
        for idx, l in enumerate(self.labels_59):
            idx_400 = self.labels_400.index(l) + 1
            label[label_400 == idx_400] = idx + 1
        label = label[np.newaxis, ...]
        return label
