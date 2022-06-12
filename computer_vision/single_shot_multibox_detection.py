import torch
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
print(batch[0].shape)


# visualize batch
grid_first_batch = make_grid(batch[0], padding=1, pad_value=0.2)
plt.figure(figsize=(9,9))
plt.imshow(grid_first_batch.permute(1, 2, 0))
plt.axis("off")
plt.show()



""" Single Shot Multibox Detection Model """



