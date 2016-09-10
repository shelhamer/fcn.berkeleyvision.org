# NYUDv2: NYU Depth Dataset V2

NYUDv2 has a curated semantic segmentation challenge with RGB-D inputs and full scene labels of objects and surfaces.
While there are many labels, we follow the 40 class task defined by

> Perceptual Organization and Recognition of Indoor Scenes from RGB-D Images.
Saurabh Gupta, Pablo Arbelaez, and Jitendra Malik.
CVPR 2013

at http://www.cs.berkeley.edu/~sgupta/pdf/GuptaArbelaezMalikCVPR13.pdf .
To reproduce the results of our paper, you must make use of the data from Gupta et al. at http://people.eecs.berkeley.edu/~sgupta/cvpr13/data.tgz .

Refer to `classes.txt` for the listing of classes in model output order.
Refer to `../nyud_layers.py` for the Python data layer for this dataset.

See the dataset site: http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html.
