import os
from turtle import forward
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
from torch.utils.data import DataLoader
import time


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

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

class Encoder():
    def __init__(self,shapeIn,nLatent):
        super(Encoder, self).__init__()
        self.shapeIn = shapeIn
        self.nLatent = nLatent

        self.enc1 = self.encoder_block(1,10,3,2,1)
        self.enc2 = self.encoder_block(10,20,3,2,1)
        self.lin1 = self.linear(980,nLatent)
        self.lin2 = self.linear(980,nLatent)

    def encoder_block(self, input_channel, output_channel, kernel_size, stride, padding):
        return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
                #nn.BatchNorm2d(output_channel),
                nn.ReLU(0.2)#,inplace=True),
            )   

    def linear(self,nIn,nOut):
        return nn.Sequential(
            nn.Linear(nIn,nOut),
            #nn.BatchNorm2d(output_channel),
            nn.ReLU(0.2)#,inplace=True)
        )

    def forward(self,z):
        #print(z.shape)
        z = self.enc1(z)
        #print(z.shape)
        z = self.enc2(z)
        #print(z.shape)
        z = z.view(-1, 7*7*20)
        #print(z.shape)
        mu = self.lin1(z)
        logVar = self.lin2(z)
        #print(mu.shape,logVar.shape)
        return mu, logVar

class Decoder():
    def __init__(self,shapeOut,nLatent):
        super(Decoder, self).__init__()

        self.linear1 = self.linear(nLatent,7*7*20)
        self.dec1 = self.decoder_block(20,10,3,3,2)
        self.dec2 = self.decoder_block(10,1,2,2,3)

    def decoder_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding),
            #nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2)#,inplace=True),
        )   

    def linear(self,nLatent,nOut):
        return nn.Sequential(
                nn.Linear(nLatent,nOut),
                #nn.BatchNorm2d(output_channel),
                nn.ReLU(0.2)#,inplace=True),
            )

    def forward(self,z):
        #print(z.shape)
        z = self.linear1(z)
        #print(z.shape)
        z = z.view(-1,20,7,7)
        z = self.dec1(z)
        #print(z.shape)
        z = self.dec2(z)
        #print(z.shape)
        return z
    
class Diffusion():
    def __init__(self,shapeIn,nLatent):
        super(Diffusion, self).__init__()

        self.encoder = Encoder(shapeIn,nLatent)
        self.decoder = Decoder(shapeIn,nLatent) 
        # not needed in first instance?
        # self.time_dimen = nn.Sequential(
        #         SinusoidalPositionEmbeddings(dim),
        #         nn.Linear(dim, time_dim),
        #         nn.GELU(),
        #         nn.Linear(time_dim, time_dim),
        #     )

    def reparameterize(self, mu, logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, z): 
        mu,logVar = self.encoder.forward(z)
        z = self.reparameterize(mu, logVar)
        return self.decoder.forward(z)
    


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
    plot_noise = False
    batch_size = 128
    epochs = 1
    learning_rate = 1e-3

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
    if plot_noise:
        plot_several(xs,ts)

    diffusion = Diffusion([28,28],100)
    
    dataloader = DataLoader(data["train"], batch_size=batch_size, shuffle=True, drop_last=True )
    optimiser = torch.Adam(dataloader.parameters(), lr=learning_rate)


    def backpropogate(epochs,dataloader,diffusion,optimiser,steps,losses):
        
        if(len(steps) != 0):
            step = steps[-1]
        else:
            step =0

        for epoch in range(epochs):
            epoch_start_time = time.time()
            for i, data in enumerate(dataloader, 0):
                data=data.to(device)
                steps.append(step)
                step += 1


    

    backpropogate(epochs,dataloader,diffusion,)
    #x_out = diffusion.forward(x).detach()
    #plot_several(torch.cat((x,x_out),0),["Original","Generated"])

if __name__ == "__main__":
    main()