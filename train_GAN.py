import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time

from import_data import import_data, prepare_label_array


class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5),
        

        )

    def forward(self, x):
        return self.main(x)

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5),
        

        )

    def forward(self, x):
        return self.main(x)



def main():
    
    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    testFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    digitClassifierModelPath = 'savedModel.pt'

    data,numbers = import_data(trainingFiles[0],trainingFiles[1])
    labels = prepare_label_array(numbers)

    epochs = 1
    learningRate = 0.1
