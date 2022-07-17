import torch
from os.path import join


BATCH_SIZE: int = 32     # number of images in a batch
NUM_EPOCHS: int = 100    # number of training epochs
RESIZE_TO : int = 250    # the new image size

# Specify the computer device {CPU/GPU}
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# DATASET 
DATAPATH : str = './dataset'
TRAIN_DIR: str = join(DATAPATH, 'train')
TEST_DIR : str = join(DATAPATH, 'test')


# CLASSES
CLASSES : list = ['background', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

NUM_CLASSES : int = len(CLASSES)

# Save the model after each 10 epochs
SAVE_INTERMEDIATE_MODEL : bool = True
SAVE_MODEL_EACH : int = 10

# The output directory
OUTPUT_DIR: str = "./output"
