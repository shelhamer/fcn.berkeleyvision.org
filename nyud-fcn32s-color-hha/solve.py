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

color_proto = '../nyud-rgb-32s/trainval.prototxt'
color_weights = '../nyud-rgb-32s/nyud-rgb-32s-28k.caffemodel'
hha_proto = '../nyud-hha-32s/trainval.prototxt'
hha_weights = '../nyud-hha-32s/nyud-hha-32s-60k.caffemodel'

# init
caffe.set_device(int(sys.argv[1]))
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')

# surgeries
color_net = caffe.Net(color_proto, color_weights, caffe.TEST)
surgery.transplant(solver.net, color_net, suffix='color')
del color_net

hha_net = caffe.Net(hha_proto, hha_weights, caffe.TEST)
surgery.transplant(solver.net, hha_net, suffix='hha')
del hha_net

interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
test = np.loadtxt('../data/nyud/test.txt', dtype=str)

for _ in range(50):
    solver.step(2000)
    score.seg_tests(solver, False, val, layer='score')
