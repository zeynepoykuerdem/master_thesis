import torch
import torch.nn as nn
import torch.cuda as cuda
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.metrics import accuracy_score

# device configuration for GPU usage
device= torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def data_augmentation():
  # Define the mean and std for normalization
  mean= (0.4914, 0.4822, 0.4465)
  std= (0.2023, 0.1994, 0.201)

  # Define transformations for training and test data
  train_transform=transforms.Compose(
    [
      #data augmentation steps for the training data
      transforms.RandomHorizontalFlip(),
      transforms.RandomCrop(32, padding=4),
      #preprocessing steps for the training data
      transforms.ToTensor(),
      transforms.Normalize(mean,std)

    ]
  )
  test_transform=transforms.Compose(
    [
      #preprocessing steps for the test data
      transforms.ToTensor(),
      transforms.Normalize(mean,std)
    ]
  )
  return train_transform,test_transform
def load_cifar10_data():
  train_transform,test_transform=data_augmentation()
  train_dataset= torchvision.datasets.CIFAR10(root='./data', train=True,transform=train_transform,download=True)

  test_dataset= torchvision.datasets.CIFAR10(root='./data', train=False,transform=test_transform,download=True)
  
  return train_dataset,test_dataset

def load_cifar10_data_no_augmentation():
  train_dataset= torchvision.datasets.CIFAR10(root='./data', train=True,transform=transforms.ToTensor(),download=True)

  test_dataset= torchvision.datasets.CIFAR10(root='./data', train=False,transform=transforms.ToTensor(),download=True)
  
  return train_dataset,test_dataset


def train_data_loader(train_dataset,batch_size):
  train_loader= DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,# shuffle training data for better generalization -> seeing the effects in memorization
    num_workers=2,
    pin_memory=True # -> faster data transfer to GPU, not supported in mps
  )
  return train_loader


def test_data_loader(test_dataset,batch_size):
  test_loader= DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,# no need to shuffle test data
    num_workers=2,
    pin_memory=True # -> faster data transfer to GPU
  )
  return test_loader




  
  
