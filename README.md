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

**PASCAL VOC models**: trained online with high momentum for a ~5 point boost in mean intersection-over-union over the original models.
These models are trained using extra data from [Hariharan et al.](http://www.cs.berkeley.edu/~bharath2/codes/SBD/download.html), but excluding SBD val.
FCN-32s is fine-tuned from the [ILSVRC-trained VGG-16 model](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014), and the finer striders are then fine-tuned in turn.
The "at-once" FCN-8s is fine-tuned from VGG-16 all-at-once by scaling the skip connections to better condition optimization.

* [FCN-32s PASCAL](voc-fcn32s): single stream, 32 pixel prediction stride net, scoring 63.6 mIU on seg11valid
* [FCN-16s PASCAL](voc-fcn16s): two stream, 16 pixel prediction stride net, scoring 65.0 mIU on seg11valid
* [FCN-8s PASCAL](voc-fcn8s): three stream, 8 pixel prediction stride net, scoring 65.5 mIU on seg11valid and 67.2 mIU on seg12test
* [FCN-8s PASCAL at-once](voc-fcn8s-atonce): all-at-once, three stream, 8 pixel prediction stride net, scoring 65.4 mIU on seg11valid

[FCN-AlexNet PASCAL](voc-fcn-alexnet): AlexNet (CaffeNet) architecture, single stream, 32 pixel prediction stride net, scoring 48.0 mIU on seg11valid.
Unlike the FCN-32/16/8s models, this network is trained with gradient accumulation, normalized loss, and standard momentum.
(Note: when both FCN-32s/FCN-VGG16 and FCN-AlexNet are trained in this same way FCN-VGG16 is far better; see Table 1 of the paper.)

To reproduce the validation scores, use the [seg11valid](https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt) split defined by the paper in footnote 7. Since SBD train and PASCAL VOC 2011 segval intersect, we only evaluate on the non-intersecting set for validation purposes.

**NYUDv2 models**: trained online with high momentum on color, depth, and HHA features (from Gupta et al. https://github.com/s-gupta/rcnn-depth).
These models demonstrate FCNs for multi-modal input.

* [FCN-32s NYUDv2 Color](nyud-fcn32s-color): single stream, 32 pixel prediction stride net on color/BGR input
* [FCN-32s NYUDv2 HHA](nyud-fcn32s-hha): single stream, 32 pixel prediction stride net on HHA input
* [FCN-32s NYUDv2 Early Color-Depth](nyud-fcn32s-color-d): single stream, 32 pixel prediction stride net on early fusion of color and (log) depth for 4-channel input
* [FCN-32s NYUDv2 Late Color-HHA](nyud-fcn32s-color-hha): single stream, 32 pixel prediction stride net by late fusion of FCN-32s NYUDv2 Color and FCN-32s NYUDv2 HHA

**SIFT Flow models**: trained online with high momentum for joint semantic class and geometric class segmentation.
These models demonstrate FCNs for multi-task output.

* [FCN-32s SIFT Flow](siftflow-fcn32s): single stream stream, 32 pixel prediction stride net
* [FCN-16s SIFT Flow](siftflow-fcn16s): two stream, 16 pixel prediction stride net
* [FCN-8s SIFT Flow](siftflow-fcn8s): three stream, 8 pixel prediction stride net

*Note*: in this release, the evaluation of the semantic classes is not quite right at the moment due to an issue with missing classes.
This will be corrected soon.
The evaluation of the geometric classes is fine.

**The following models have not yet been ported to master and trained with the latest settings. Check back soon.**

PASCAL-Context models including architecture definition, solver configuration, and bare-bones solving script (fine-tuned from the ILSVRC-trained VGG-16 model):
* [FCN-32s PASCAL-Context](https://gist.github.com/shelhamer/80667189b218ad570e82#file-readme-md): single stream, 32 pixel prediction stride version
* [FCN-16s PASCAL-Context](https://gist.github.com/shelhamer/08652f2ba191f64e619a#file-readme-md): two stream, 16 pixel prediction stride version
* [FCN-8s PASCAL-Context](https://gist.github.com/shelhamer/91eece041c19ff8968ee#file-readme-md): three stream, 8 pixel prediction stride version
