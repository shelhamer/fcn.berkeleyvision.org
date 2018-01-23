import numpy as np
from PIL import Image

import caffe
import vis

# the demo image is "2007_000129" from PASCAL VOC

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('demo/image.jpg')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)

# visualize segmentation in PASCAL VOC colors
voc_palette = vis.make_palette(21)
out_im = Image.fromarray(vis.color_seg(out, voc_palette))
out_im.save('demo/output.png')
masked_im = Image.fromarray(vis.vis_seg(im, out, voc_palette))
masked_im.save('demo/visualization.jpg')
