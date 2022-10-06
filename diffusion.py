import os
import torch
from torch import nn
from os.path import exists
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils
from models import Generator, Classifier
import math
import torch.nn.functional as F
import random
from import_data import import_data
from view_data import plot_several


def beta_schedule(T,s=0.008):
    # from https://arxiv.org/pdf/2102.09672.pdf pg 4
    steps = T + 1
    x = torch.linspace(0,T,steps)
    alpha = torch.cos(0.5*math.pi*((x/T)+s)/(1+s))**2
    alpha = alpha/alpha[0]
    beta = 1 - (alpha[1:]/(alpha[:-1]))
    return torch.clamp(beta,0.0001,0.9999)
    
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    t = t.type(torch.int64)
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start) #Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    #print(sqrt_alphas_cumprod_t,sqrt_one_minus_alphas_cumprod_t)
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    predicted_noise = denoise_model(x_noisy, t)

    loss = F.smooth_l1_loss(noise, predicted_noise) #Creates a criterion that uses a squared term if the absolute element-wise error falls below beta and an L1 term otherwise. It is less sensitive to outliers than torch.nn.MSELoss and in some cases prevents exploding gradients (e.g. see the paper Fast R-CNN by Ross Girshick).
    return loss

class Diffusion():
    def __init__(self,shapeIn):
        super(Generator, self).__init__()

        self.nLatent = nLatent
        self. z_dim = 100
        self.conv1 = self.get_generator_block(self.z_dim, nLatent * 4, kernel_size=3, stride=2)
        self.conv2 = self.get_generator_block(nLatent * 4, nLatent * 2, kernel_size=4, stride = 1)
        self.conv3 = self.get_generator_block(nLatent * 2, nLatent, kernel_size=3, stride = 2)
        self.convFinal = self.get_generator_final_block(nLatent, 1, kernel_size=4, stride=2)


    def get_generator_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return nn.Sequential(
                nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channel),
                nn.LeakyReLU(0.2,inplace=True),
        )    

    def get_generator_final_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return  nn.Sequential(
                nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
                nn.Tanh()
            )


    def forward(self, noise): 

        return x
    


def main():
    
    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    modelPath = 'models/diffusion.pt'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    data,numbers = import_data(trainingFiles[0],trainingFiles[1])

    T = 200
    betas = beta_schedule(T,0.008)
    alphas = 1.-betas
    alphas_cumprod = torch.cumprod(alphas,-1) # cumprod is all the products of previous elements
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    i = random.randint(0,60000)
    x = data[i:i+1]
    xs = x
    ts = np.linspace(0,199,5,dtype = int)
    for t in ts[1:]:
        x_noise = q_sample(x, torch.tensor([t]),sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod, noise=None)
        xs = torch.cat((xs,x_noise),0)

    plot_several(xs,ts)




if __name__ == "__main__":
    main()