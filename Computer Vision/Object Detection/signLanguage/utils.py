import cv2
import numpy as np
from config import DEVICE, CLASSES, RESIZE_TO
import albumentations as A
import matplotlib.pyplot as plt
import torchvision.transforms as T
from albumentations.pytorch.transforms import ToTensorV2

class Utils:
    """
    Useful functions.
    """
    def __init__(self) -> None:
        pass

    def train_transforms():
        """The train transformations.

        :return: Returns a composition of data augmentation transformations.
        :rtype: Albumentation.Compose
        """
        transforms = [
            A.Resize(RESIZE_TO, RESIZE_TO),
            A.Flip(p=0.7),
            A.RandomRotate90(p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            ToTensorV2()
            ]
        return A.Compose(transforms, bbox_params= A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def test_transforms():
        """The test transformations.

        :return: Returns a composition of data augmentation transformations.
        :rtype: Albumentation.Compose
        """
        transforms = [
            A.Resize(RESIZE_TO, RESIZE_TO),
            ToTensorV2()
            ]
        return A.Compose(transforms, bbox_params= A.BboxParams(format='pascal_voc', label_fields=['labels']))


    def visualize_data(data_loader, num = 5):
        """Visualize some dataset images. 

        :param dataset: The data loader object
        :type dataset: torch.utils.data.dataloader.DataLoader
        :param num: Number of images to visualize.
        :type num: int
        """
        images, targets = next(iter(data_loader))

        for i in range(num):
                # get the box
                box = targets['boxes'][i]
                box = np.squeeze(box.to(DEVICE).numpy()).astype(int)
                xmin, ymin, xmax, ymax = box
                box_width, box_height = xmax - xmin, ymax - ymin

                # get label
                label_idx = int(targets['labels'][i].item())
                label = CLASSES[label_idx]
                
                # get the image
                image = images[i]
                image = image.to(DEVICE)
                image = T.ToPILImage()(image)
                width, height = image.size
                image = np.array(image)

                # draw the box
                image = cv2.rectangle(image, [xmin, ymin], [xmax, ymax], color=(0,200,0), thickness=3)

                # write the label to the image
                label_position = (box[0] + box_width // 2 , box[1] + box_height // 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                image = cv2.putText(image, label, label_position, fontFace = font, fontScale=2, color = (0,0,0), thickness = 2)
                
                # visualize the image
                plt.imshow(image)
                plt.show()


class Averager:
    """
    Keep track of the training and validation loss for each epoch.
    """
    def __init__(self) -> None:
        """
        Initialize averager.
        """
        self.iterations = 0.0
        self.current_total = 0.0

    def send(self, value):
        """Add loss value to the current total value.

        :param value: loss value to add
        :type value: float
        """
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        """The epoch's loss.

        :return: loss
        :rtype: float
        """
        if self.current_total == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        """
        Reset the averager.
        """
        self.iterations = 0.0
        self.current_total = 0.0



