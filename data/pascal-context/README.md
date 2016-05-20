# PASCAL-Context

PASCAL-Context is a full object and scene labeling of PASCAL VOC 2010.
It includes both object (cat, dog, ...) and surface (sky, grass, ...) classes.

We follow the 59 class task defined by

> The Role of Context for Object Detection and Semantic Segmentation in the Wild.
Roozbeh Mottaghi, Xianjie Chen, Xiaobai Liu, Nam-Gyu Cho, Seong-Whan Lee, Sanja Fidler, Raquel Urtasun, and Alan Yuille.
CVPR 2014

which selects the 59 most common classes for learning and evaluation.

Refer to `classes-59.txt` for the listing of classes in model output order.
Refer to `../pascalcontext_layers.py` for the Python data layer for this dataset.

Note that care must be taken to map the raw class annotations into the 59 class task, as handled by our data layer.

See the dataset site: http://www.cs.stanford.edu/~roozbeh/pascal-context/
