import os
import glob
import numpy as np
import scipy.io

from PIL import Image

dataset_dir = './sbdd'

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in xrange(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette
palette = make_palette(256).reshape(-1)

for kind in ('cls', 'inst'):
    # collect the inputs
    paths = glob.glob('{}/{}/*.mat'.format(dataset_dir, kind))
    ids = [os.path.basename(p)[:-4] for p in paths]
    for i, idx in enumerate(ids):
        if i % 100 == 0:
            print "Converting {}th annotation...".format(i)
        # loading the label
        mat = scipy.io.loadmat('{}/{}/{}.mat'.format(dataset_dir, kind, idx))
        label_arr = mat['GT{}'.format(kind)][0]['Segmentation'][0].astype(np.uint8)
        # saving the label
        label_im = Image.fromarray(label_arr)
        label_im.putpalette(palette)
        label_im.save('{}/{}/{}.png'.format(dataset_dir, kind, idx))
