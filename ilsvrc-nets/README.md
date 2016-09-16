# ILSVRC Networks

These classification networks are trained on ILSVRC for object recognition.
We cast these nets into fully convolutional form to make use of their parameters as pre-training.

To reproduce our FCNs, or train your own on your own data, you need to first collect the corresponding base network.

- [VGG16](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md)
- [CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)
- [BVLC GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
