# -*- coding: utf-8 -*-
"""Polawat_Srichana_20696984 (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LINMXHfhW90i9dDvY0Vbh7WOKPj2v-zI
"""

!pip install -U d2l





!pip install torchsummary

# Commented out IPython magic to ensure Python compatibility.
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import optim
import numpy as np
# %matplotlib inline
import os
from d2l import torch as d2l
import numpy as np
import pandas as pd
from torchsummary import summary

from google.colab import drive
drive.mount('/content/gdrive')

model_save_name = 'polawat.pt'
path = F"/content/gdrive/MyDrive/{model_save_name}"
net = torchvision.models.resnet18()
net.state_dict()
torch.save(net.state_dict(), path)

model_save_name = 'polawat.pt'
path = F"/content/gdrive/MyDrive/{model_save_name}"
net = torchvision.models.resnet18()
net.load_state_dict(torch.load(path))

normalize = torchvision.transforms.Normalize(
   [0.1307], [0.3081])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])
trainset =torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=train_augs, download=True)
testset=torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=test_augs, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=1,
                      param_group=True):
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, trainloader, testloader, loss, trainer, num_epochs,
                   devices)

Fine_tuning =train_fine_tuning(net, 5e-5)

net = torchvision.models.resnet18()
net.state_dict()

"""# Training from Scratch (ResNet18)"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 10))

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.05, 5, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
Model = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())

"""# Fine-tuning (ResNet18)"""

net = torchvision.models.resnet18(pretrained=True)

resnet18= torchvision.models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 10)
nn.init.xavier_uniform_(resnet18.fc.weight);

X = torch.rand(size=(224, 224))
X.unsqueeze_(0)
X=X.repeat(3,1,1)
X.shape

normalize = torchvision.transforms.Normalize(
   [0.1307], [0.3081])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])
trainset =torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=train_augs, download=True)
testset=torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=test_augs, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

def train_fine_tuning(resnet18, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in resnet18.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': resnet18.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(resnet18.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(resnet18, trainloader, testloader, loss, trainer, num_epochs,
                   devices)

Fine_tuning =train_fine_tuning(resnet18, 5e-5)

"""# Fine Tuning(ResNet50)"""

import gc
import torch
torch.cuda.empty_cache()

net = torchvision.models.resnet50(pretrained=True)

net.fc

resnet50 = torchvision.models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 10)
nn.init.xavier_uniform_(resnet50.fc.weight);

X = torch.rand(size=(224, 224))
X.unsqueeze_(0)
X=X.repeat(3,1,1)
X.shape

normalize = torchvision.transforms.Normalize(
   [0.1307], [0.3081])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])
trainset =torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=train_augs, download=True)
testset=torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=test_augs, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

def train_fine_tuning(resnet50 , learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in resnet50 .named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': resnet50 .fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(resnet50 .parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(resnet50 , trainloader, testloader, loss, trainer, num_epochs,
                   devices)

train_fine_tuning(resnet50 , 5e-5)

"""# Fine Tuning(Alexnet)

"""

net = torchvision.models.alexnet(pretrained=True)

alex_net = torchvision.models.alexnet(pretrained=True)
alex_net.classifier[6]=nn.Linear(4096,10)

alex_net

X = torch.rand(size=(224, 224))
X.unsqueeze_(0)
X=X.repeat(3,1,1)
X.shape

normalize = torchvision.transforms.Normalize(
   [0.1307], [0.3081])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])
trainset =torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=train_augs, download=True)
testset=torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=test_augs, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

def train_fine_tuning(alex_net, learning_rate, batch_size=128, num_epochs=10,):
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(alex_net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(alex_net, trainloader, testloader, loss, trainer, num_epochs,
                   devices)

train_fine_tuning(alex_net, 5e-5)

"""# Task3

# Original Input(With Softmax)
"""

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0.1307, std=0.3081)

net.apply(init_weights);

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

"""#Last convolutional layer"""

finetune_net = torchvision.models.resnet18(pretrained=True)
for param in finetune_net.layer1.parameters():
    param.requires_grad = False
for param in finetune_net.layer2.parameters():
    param.requires_grad = False
for param in finetune_net.layer3.parameters():
    param.requires_grad = False
for param in finetune_net.conv1.parameters():
    param.requires_grad = False
for param in finetune_net.bn1.parameters():
    param.requires_grad = False
for param in finetune_net.fc.parameters():
    param.requires_grad = False

net=nn.Sequential(finetune_net,nn.LogSoftmax(1))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0.1307, std=0.3081)
net.apply(init_weights);
net.cuda()

summary(net,(3,224,224))

normalize = torchvision.transforms.Normalize(
   [0.1307], [0.3081])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])
trainset =torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=train_augs, download=True)
testset=torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=test_augs, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

def train_fine_tuning(net, learning_rate, batch_size=256, num_epochs=5,
                      param_group=True):
    devices = d2l.try_all_gpus()
    loss = nn.NLLLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, trainloader, testloader, loss, trainer, num_epochs,
                   devices)

train_fine_tuning(net, 5e-4)

"""#Intermediate convolutional layer"""

finetune_net = torchvision.models.resnet18(pretrained=True)
for param in finetune_net.layer1.parameters():
    param.requires_grad = False
for param in finetune_net.layer2.parameters():
    param.requires_grad = False
for param in finetune_net.layer4.parameters():
    param.requires_grad = False
for param in finetune_net.conv1.parameters():
    param.requires_grad = False
for param in finetune_net.bn1.parameters():
    param.requires_grad = False
for param in finetune_net.fc.parameters():
    param.requires_grad = False

net1=nn.Sequential(finetune_net,nn.LogSoftmax(1))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0.1307, std=0.3081)
net1.apply(init_weights);
net1.cuda()

summary(net1,(3,224,224))

normalize = torchvision.transforms.Normalize(
   [0.1307], [0.3081])
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Lambda(lambda X:X.repeat(3,1,1)),
    normalize])
trainset =torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=True, transform=train_augs, download=True)
testset=torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', train=False, transform=test_augs, download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

def train_fine_tuning(net1, learning_rate, batch_size=256, num_epochs=5,
                      param_group=True):
    devices = d2l.try_all_gpus()
    loss = nn.NLLLoss(reduction="none")
    trainer = torch.optim.SGD(net1.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net1, trainloader, testloader, loss, trainer, num_epochs,
                   devices)

train_fine_tuning(net1, 5e-4)