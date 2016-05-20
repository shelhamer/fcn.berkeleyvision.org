# SIFT Flow

SIFT Flow is a semantic segmentation dataset with two labelings:

- semantic classes, such as "cat" or "dog"
- geometric classes, consisting of "horizontal, vertical, and sky"

Refer to `classes.txt` for the listing of classes in model output order.
Refer to `../siftflow_layers.py` for the Python data layer for this dataset.

Note that the dataset has a number of issues, including unannotated images and missing classes from the test set.
The provided splits exclude the unannotated images.
As noted in the paper, care must be taken for proper evalution by excluding the missing classes.

Download the dataset:
http://www.cs.unc.edu/~jtighe/Papers/ECCV10/siftflow/SiftFlowDataset.zip
