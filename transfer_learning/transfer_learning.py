import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import copy

from Dataset_bees_ants import load_dataset
from training import training_model

print()
print("Transfer learning with pytorch")
print("------------------------------")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device using: {device}")
print()


###################
# data augmentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'hymenoptera_data'

image_datasets, dataloaders, dataset_size, class_names = load_dataset(data_dir, 
                                                                      data_transforms)


print("Datasets")
print("--------")
print(image_datasets)
print()
print(f"Class names: {class_names}")


########################
# visualize a few images

def imshow(inp, title=None):
    """ imshow for tensor """
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean 
    inp = np.clip(inp,0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


# get batch of training data
inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])





##################
# Fine Tuning 
# pretrained model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)



##################
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# scheduler
# every 7. epoch lr is multiplied by gamma
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

print("Finetuning model")
print("----------------")

model = training_model(model, dataloaders, dataset_size, criterion, 
                       optimizer, step_lr_scheduler, device, num_epochs=5)




#########################
# Freeze all layers in the beginning and only training
# the last "new" layer

model_2 = models.resnet18(pretrained=True)
for param in model_2.parameters():
    param.requires_grad = False

num_features = model_2.fc.in_features
model_2.fc = nn.Linear(num_features, 2)

model_2.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_2.parameters(), lr=0.001)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


print("Training last layer")
print("-------------------")
model_2 = training_model(model_2, dataloaders, dataset_size, criterion, 
                         optimizer, step_lr_scheduler, device, num_epochs=5)




# look at the results
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


visualize_model(model)
visualize_model(model_2)


