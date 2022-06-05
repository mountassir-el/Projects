from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteHead
from functools import partial
import torch.nn as nn
from config import NUM_CLASSES

def create_model(num_classes):
    """Create and return the detection model.

    Args:
        num_classes (int): the number of classes to predict with the model.

    Returns:
        torch.nn.Module: the detection model
    """

    # create the model
    model = ssdlite320_mobilenet_v3_large(pretrained=True)

    # get the details needed to change the model's head
    size = (320, 320)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    out_channels = det_utils.retrieve_out_channels(model.backbone, size)
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    # change the prediction layers
    model.head = SSDLiteHead(out_channels, num_anchors, NUM_CLASSES, norm_layer)

    return model