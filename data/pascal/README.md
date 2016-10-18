# PASCAL VOC and SBD

PASCAL VOC is a standard recognition dataset and benchmark with detection and semantic segmentation challenges.
The semantic segmentation challenge annotates 20 object classes and background.
The Semantic Boundary Dataset (SBD) is a further annotation of the PASCAL VOC data that provides more semantic segmentation and instance segmentation masks.

PASCAL VOC has a private test set and [leaderboard for semantic segmentation](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6).

The train/val/test splits of PASCAL VOC segmentation challenge and SBD diverge.
Most notably VOC 2011 segval intersects with SBD train.
Care must be taken for proper evaluation by excluding images from the train or val splits.

We train on the 8,498 images of SBD train.
We validate on the non-intersecting set defined in the included `seg11valid.txt`.

Refer to `classes.txt` for the listing of classes in model output order.
Refer to `../voc_layers.py` for the Python data layer for this dataset.

See the dataset sites for download:

- PASCAL VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
- SBD: see [homepage](http://home.bharathh.info/home/sbd) or [direct download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)
