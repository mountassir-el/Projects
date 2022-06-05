import torch
from os.path import join


BATCH_SIZE:int = 32
NUM_EPOCHS:int = 200
RESIZE_TO:int = 250

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

DATAPATH = './dataset'
TRAIN_DIR = join(DATAPATH, 'train')
VALID_DIR = join(DATAPATH, 'valid')
TEST_DIR  = join(DATAPATH, 'test')

CLASSES = ['background', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

NUM_CLASSES = len(CLASSES)

SAVE_INTERMEDIATE_MODEL:bool = True
SAVE_MODEL_EACH:int = 10