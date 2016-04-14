These are models and scripts for the [paper](http://www.cs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf):

    Fully Convolutional Models for Semantic Segmentation
    Jonathan Long*, Evan Shelhamer*, Trevor Darrell
    CVPR 2015
    arXiv:1411.4038

and its journal edition in PAMI (to appear).

**Note that this is a work in progress and the final, reference version is coming soon.**
Please ask Caffe and FCN usage questions on the [caffe-users mailing list](https://groups.google.com/forum/#!forum/caffe-users).

These models are compatible with `BVLC/caffe:master` @ 8c66fa5 with the merge of PRs BVLC/caffe#3613 and BVLC/caffe#3570.
The code and models here are available under the same license as Caffe (BSD-2) and the Caffe-bundled models (that is, unrestricted use; see the [BVLC model license](http://caffe.berkeleyvision.org/model_zoo.html#bvlc-model-license)).

PASCAL VOC models: trained online with high momentum for a ~5 point boost in mean intersection-over-union over the original models.
These models are trained using extra data from [Hariharan et al.](http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html), but excluding SBD val.
FCN-32s is fine-tuned from the [ILSVRC-trained VGG-16 model](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014), and the finer striders are then fine-tuned in turn.

* [FCN-32s PASCAL](tree/master/fcn32s): single stream, 32 pixel prediction stride version, scoring 63.6 mIU on seg11valid
* [FCN-16s PASCAL](tree/master/fcn16s): two stream, 16 pixel prediction stride version, scoring 65.0 mIU on seg11valid
* [FCN-8s PASCAL](tree/master/fcn8s): three stream, 8 pixel prediction stride version, scoring 65.5 mIU on seg11valid and 67.2 mIU on seg12test

To reproduce the validation scores, use the [seg11valid](https://gist.github.com/shelhamer/edb330760338892d511e) split defined by the paper in footnote 7. Since SBD train and PASCAL VOC 11 segval intersect, we only evaluate on the non-intersecting set for validation purposes.

**The following models have not yet been ported to master and trained with the latest settings. Check back soon.**

PASCAL VOC:
* [FCN-AlexNet PASCAL](https://gist.github.com/shelhamer/3f2c75f3c8c71357f24c#file-readme.md): AlexNet (CaffeNet) single stream, 32 pixel prediction stride version

SIFT Flow model (also fine-tuned from VGG-16):
* [FCN-16s SIFT Flow](https://gist.github.com/longjon/f35e3a101e1478f721f5#file-readme-md): two stream, 16 pixel prediction stride version

NYUDv2 models (also fine-tuned from VGG-16, and using HHA features from Gupta et al. https://github.com/s-gupta/rcnn-depth):
* [FCN-32s NYUDv2](https://gist.github.com/longjon/16db1e4ad3afc2614067#file-readme-md): single stream, 32 pixel prediction stride version
* [FCN-16s NYUDv2](https://gist.github.com/longjon/dd1f5097af6b531bddcc#file-readme-md): two stream, 16 pixel prediction stride version

PASCAL-Context models including architecture definition, solver configuration, and bare-bones solving script (fine-tuned from the ILSVRC-trained VGG-16 model):
* [FCN-32s PASCAL-Context](https://gist.github.com/shelhamer/80667189b218ad570e82#file-readme-md): single stream, 32 pixel prediction stride version
* [FCN-16s PASCAL-Context](https://gist.github.com/shelhamer/08652f2ba191f64e619a#file-readme-md): two stream, 16 pixel prediction stride version
* [FCN-8s PASCAL-Context](https://gist.github.com/shelhamer/91eece041c19ff8968ee#file-readme-md): three stream, 8 pixel prediction stride version
