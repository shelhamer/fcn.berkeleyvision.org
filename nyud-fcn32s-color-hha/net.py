import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def modality_fcn(net_spec, data, modality):
    n = net_spec
    # the base net
    n['conv1_1' + modality], n['relu1_1' + modality] = conv_relu(n[data], 64,
                                                                 pad=100)
    n['conv1_2' + modality], n['relu1_2' + modality] = conv_relu(n['relu1_1' +
        modality], 64)
    n['pool1' + modality] = max_pool(n['relu1_2' + modality])

    n['conv2_1' + modality], n['relu2_1' + modality] = conv_relu(n['pool1' +
        modality], 128)
    n['conv2_2' + modality], n['relu2_2' + modality] = conv_relu(n['relu2_1' +
        modality], 128)
    n['pool2' + modality] = max_pool(n['relu2_2' + modality])

    n['conv3_1' + modality], n['relu3_1' + modality] = conv_relu(n['pool2' +
        modality], 256)
    n['conv3_2' + modality], n['relu3_2' + modality] = conv_relu(n['relu3_1' +
        modality], 256)
    n['conv3_3' + modality], n['relu3_3' + modality] = conv_relu(n['relu3_2' +
        modality], 256)
    n['pool3' + modality] = max_pool(n['relu3_3' + modality])

    n['conv4_1' + modality], n['relu4_1' + modality] = conv_relu(n['pool3' +
        modality], 512)
    n['conv4_2' + modality], n['relu4_2' + modality] = conv_relu(n['relu4_1' +
        modality], 512)
    n['conv4_3' + modality], n['relu4_3' + modality] = conv_relu(n['relu4_2' +
        modality], 512)
    n['pool4' + modality] = max_pool(n['relu4_3' + modality])

    n['conv5_1' + modality], n['relu5_1' + modality] = conv_relu(n['pool4' +
        modality], 512)
    n['conv5_2' + modality], n['relu5_2' + modality] = conv_relu(n['relu5_1' +
        modality], 512)
    n['conv5_3' + modality], n['relu5_3' + modality] = conv_relu(n['relu5_2' +
        modality], 512)
    n['pool5' + modality] = max_pool(n['relu5_3' + modality])

    # fully conv
    n['fc6' + modality], n['relu6' + modality] = conv_relu(
        n['pool5' + modality], 4096, ks=7, pad=0)
    n['drop6' + modality] = L.Dropout(
        n['relu6' + modality], dropout_ratio=0.5, in_place=True)
    n['fc7' + modality], n['relu7' + modality] = conv_relu(
        n['drop6' + modality], 4096, ks=1, pad=0)
    n['drop7' + modality] = L.Dropout(
        n['relu7' + modality], dropout_ratio=0.5, in_place=True)
    n['score_fr' + modality] = L.Convolution(
        n['drop7' + modality], num_output=40, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return n

def fcn(split, tops):
    n = caffe.NetSpec()
    n.color, n.hha, n.label = L.Python(module='nyud_layers',
            layer='NYUDSegDataLayer', ntop=3,
            param_str=str(dict(nyud_dir='../data/nyud', split=split,
                tops=tops, seed=1337)))
    n = modality_fcn(n, 'color', 'color')
    n = modality_fcn(n, 'hha', 'hha')
    n.score_fused = L.Eltwise(n.score_frcolor, n.score_frhha,
            operation=P.Eltwise.SUM, coeff=[0.5, 0.5])
    n.upscore = L.Deconvolution(n.score_fused,
        convolution_param=dict(num_output=40, kernel_size=64, stride=32,
            bias_term=False),
        param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.color)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))
    return n.to_proto()

def make_net():
    tops = ['color', 'hha', 'label']
    with open('trainval.prototxt', 'w') as f:
        f.write(str(fcn('trainval', tops)))

    with open('test.prototxt', 'w') as f:
        f.write(str(fcn('test', tops)))

if __name__ == '__main__':
    make_net()
