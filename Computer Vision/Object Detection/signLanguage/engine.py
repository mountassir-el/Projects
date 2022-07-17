import torch.optim as optim
from config import DEVICE, OUTPUT_DIR, SAVE_INTERMEDIATE_MODEL, SAVE_MODEL_EACH
from utils import Averager
from tqdm import tqdm
from torch import save
from os.path import join


def train(model, data_loader, num_epochs, early_stopping = False, patience = 10):
    """train the model.

    :param model: the torch model.
    :type model: torch.nn.Module
    :param data_loader: the train data loader
    :type data_loader: torch.utils.data.DataLoader
    :param num_epochs: number of epochs to train the model.
    :type num_epochs: int
    :param early_stopping: Whether to apply early stopping or not, defaults to False
    :type early_stopping: bool, optional
    :param patience: number of epochs to wait when applying early stopping, defaults to 10
    :type patience: int, optional
    :return: the training loss values 
    :rtype: list
    """
    # Early stopping trigger
    triggertimes = 0

    # Load model to device
    model = model.to(DEVICE)

    # Specify optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params = params, lr=0.1, momentum=0.9)

    # training losses
    train_loss_list = []


    epochLossAverager = Averager()
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

            epochLossAverager.send(loss_value)

            # update progress bar
            prog_bar.set_description(f"Epoch [{epoch_idx}/{num_epochs}]")
            prog_bar.set_postfix(loss = epochLossAverager.value)

        
        epoch_loss = epochLossAverager.value

        if early_stopping and epoch_idx > 1:
            # apply early stopping
            last_loss = train_loss_list[-1]
            if epoch_loss >= last_loss:
                triggertimes += 1
            else:
                triggertimes = 0

            if triggertimes >= patience:
                return train_loss_list
            

        # save intermediate models
        if SAVE_INTERMEDIATE_MODEL:
            if epoch_idx % SAVE_MODEL_EACH == 0:
                model_path = join(OUTPUT_DIR, "model_" + str(epoch_idx) + ".pth")
                save(model, model_path)
        
        train_loss_list.append(epoch_loss)
        epochLossAverager.reset()
    
    return train_loss_list
