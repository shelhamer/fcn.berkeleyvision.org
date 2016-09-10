import os
import copy
import glob
import numpy as np

from PIL import Image


class voc:
    def __init__(self, data_path):
        # data_path is /path/to/PASCAL/VOC2011
        self.dir = data_path
        self.classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person',
                        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        # for paletting
        reference_idx = '2008_000666'
        palette_im = Image.open('{}/SegmentationClass/{}.png'.format(
            self.dir, reference_idx))
        self.palette = palette_im.palette

    def load_image(self, idx):
        im = Image.open('{}/JPEGImages/{}.jpg'.format(self.dir, idx))
        return im

    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label = Image.open('{}/SegmentationClass/{}.png'.format(self.dir, idx))
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

    def palette(self, label_im):
        '''
        Transfer the VOC color palette to an output mask for visualization.
        '''
        if label_im.ndim == 3:
            label_im = label_im[0]
        label = Image.fromarray(label_im, mode='P')
        label.palette = copy.copy(self.palette)
        return label
