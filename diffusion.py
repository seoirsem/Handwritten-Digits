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

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

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


    x = data[110:111]
    xs = x
    ts = np.linspace(0,4,5,dtype = int)
    for t in ts[1:]:
        x_noise = q_sample(x, torch.tensor([t]),sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod, noise=None)
        xs = torch.cat((xs,x_noise),0)

    plot_several(xs,ts)




if __name__ == "__main__":
    main()