import os 


# DATASET DETAILS
NB_IMAGES = 3000
NB_IMAGES_TRAIN = 2300
NB_IMAGES_TEST = 700

# FOLDERS
IMAGE_FOLDER : str = "images"
ANNOTATION_FOLDER : str = "annotations"
TEST : str = "test"
TRAIN : str = "train"

# DATA PATHS
RAW_DATAPATH : str = "./raw_dataset"
DATAPATH : str = "dataset"
TRAIN_DATAPATH : str = os.path.join(DATAPATH, TRAIN)
TEST_DATAPATH : str = os.path.join(DATAPATH, TEST)