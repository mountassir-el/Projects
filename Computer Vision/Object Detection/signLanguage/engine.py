import torch.optim as optim
from config import DEVICE, SAVE_INTERMEDIATE_MODEL, SAVE_MODEL_EACH
from utils import Averager
from tqdm import tqdm
from torch import save


def train(model, data_loader, num_epochs = 10):
    model = model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params = params, lr=0.1, momentum=0.9)

    train_loss_list = []
    epoch_loss = Averager()
    for epoch_idx in range(1,num_epochs+1):
        prog_bar = tqdm(enumerate(data_loader), total=len(data_loader), leave = False)
        for i, (images, targets) in prog_bar:
            current_batch_size = len(images)
            images = [image.to(DEVICE) for image in images]
            targets = [{k: targets[k][idx].to(DEVICE) for k in targets.keys()} for idx in range(current_batch_size)]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            loss_value = total_loss.item()
            total_loss.backward()
            optimizer.step()

            epoch_loss.send(loss_value)

            # update progress bar
            prog_bar.set_description(f"Epoch [{epoch_idx}/{num_epochs}]")
            prog_bar.set_postfix(loss = epoch_loss.value)

        if SAVE_INTERMEDIATE_MODEL:
            if epoch_idx % SAVE_MODEL_EACH == 0:
                save(model, "model_" + str(epoch_idx) + ".pth")
        
        train_loss_list.append(epoch_loss.value)
        epoch_loss.reset()
    
    return train_loss_list
