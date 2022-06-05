from model import create_model
from config import NUM_CLASSES


model = create_model(NUM_CLASSES)


from datasets import SingLanguageDataset
from torch.utils.data import DataLoader
from config import TRAIN_DIR, TEST_DIR, VALID_DIR
from utils import Utils


train_ds = SingLanguageDataset(TRAIN_DIR, Utils.get_transforms(train=True))
train_loader = DataLoader(train_ds, batch_size= 20, shuffle= True, num_workers= 0)

valid_ds = SingLanguageDataset(VALID_DIR)
valid_loader = DataLoader(valid_ds, batch_size= 20, shuffle= False, num_workers= 0)

test_ds = SingLanguageDataset(TEST_DIR)
test_loader = DataLoader(test_ds, batch_size= 20, shuffle= False, num_workers= 0)


import torch.optim as optim
from config import DEVICE
from utils import Averager

def train_one_epoch(model, data_loader):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params = params, lr=0.001, momentum=0.9)
    epoch_loss = Averager()

    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        images, targets = data
        images = [image.to(DEVICE) for image in images]
        current_batch_size = len(images)
        targets = [{k: targets[k][idx].to(DEVICE) for k in targets.keys()} for idx in range(current_batch_size)]

        print('calculating loss ...')
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss.send(loss_value)
        print('loss value : ', loss_value, ' ', epoch_loss.value)
        losses.backward()
        optimizer.step()
        

    return epoch_loss.value


train_one_epoch(model, train_loader)