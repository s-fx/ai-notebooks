import torch
from torchvision import datasets, transforms
import os



""" Folder structure has to be like
    - data
        - train
            - bees
            - ants
        - val
            - bees
            - ants
"""



def load_dataset(img_dir, transforms):
    image_datasets = {x: datasets.ImageFolder(os.path.join(img_dir, x),
                                              transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                      batch_size=4,
                                                      shuffle=True,
                                                      num_workers=4)
                   for x in ['train', 'val']}
    dataset_size = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return image_datasets, dataloaders, dataset_size, class_names


