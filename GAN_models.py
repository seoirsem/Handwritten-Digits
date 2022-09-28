import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time


class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        ngf = 32
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d( 100, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
            )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
            )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d( ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True)
            )
        self.conv4 = nn.Sequential(    
            nn.ConvTranspose2d( ngf * 1, 1, 4, 2, 3, bias=False),
            nn.Tanh()
        )
        

    def forward(self, x):
        #print(x.shape)
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)
        return x

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 32
        kernalSize = 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, ndf, kernalSize, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernalSize, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
            # state size. (ndf*2) x 16 x 16
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernalSize, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, kernalSize, 2, 1, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
#        print(x.shape)
        x = self.conv1(x)
#        print(x.shape)
        x = self.conv2(x)
#        print(x.shape)
        x = self.conv3(x)
#        print(x.shape)
        x = self.conv4(x)
#        print(x.shape)

        return x#self.main(x)