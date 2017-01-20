import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
base_net = caffe.Net('../ilsvrc-nets/vgg16fcn.prototxt', '../vgg16fc.caffemodel',
        caffe.TEST)

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
surgery.transplant(solver.net, base_net)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

solver.net.params['conv1_1_bgrd'][0].data[:, :3] = base_net.params['conv1_1'][0].data
solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.mean(base_net.params['conv1_1'][0].data, axis=1)
solver.net.params['conv1_1_bgrd'][1].data[...] = base_net.params['conv1_1'][1].data

del base_net

# scoring
test = np.loadtxt('../data/nyud/test.txt', dtype=str)

for _ in range(50):
    solver.step(2000)
    score.seg_tests(solver, False, val, layer='score')
