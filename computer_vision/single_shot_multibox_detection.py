import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from object_detection_dataset import BananasDataset




batch_size, edge_size = 32, 256


""" Loading the dataset """
dataset_train = BananasDataset(is_train=True)
dataset_val = BananasDataset(is_train=False)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)


batch = next(iter(train_loader))
print(f"Batch shape: {batch[0].shape}")


# visualize batch
grid_first_batch = make_grid(batch[0], padding=1, pad_value=0.2)
plt.figure(figsize=(9,9))
plt.imshow(grid_first_batch.permute(1, 2, 0))
plt.axis("off")
plt.show()



""" Single Shot Multibox Detection Model
    Base Network (deep CNN) and several multiscale feature map blocks
    generates varying number of anchor boxes with different sizes and detects
    varying size objects by predicting classes and offsets of these anchor
    boxes """

# Class Prediction Layer
# output = predicting nb of object classes + 1 (background)
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)



# Bounding Box Prediction Layer
# number of outputs = predict four offsets
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)



def forward(x, block):
    return block(x)


Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
print(Y1.shape)
print(Y2.shape)


def flatten_preds(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_preds(p) for p in preds], dim=1)

print(concat_preds([Y1, Y2]).shape)



""" downsampling block
    to detect objects at multiple scales
    3 x 3 conv layers with padding=1 do not change the shape
    of feature maps.
    2 x 2 max pool layer reduces h and w of input feature maps by half """
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLu())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)



""" base network block
    extracts features from input images. """
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) -1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


""" the complete single shot multibox detection model consists of 5 blocks.
    






