# Loading the necessary modules and dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F

# Load the CIFAR10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 50000 training images and 10000 test images
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# split the 50000 training images into 40000 training and 10000 shadow
train_dataset, shadow_dataset = random_split(trainset, [40000, 10000])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
shadow_loader = DataLoader(shadow_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=True)
