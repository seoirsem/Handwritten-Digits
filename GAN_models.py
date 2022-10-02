import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
from torch import autograd

class Generator(nn.Module):
    
    def __init__(self,nLatent):
        super(Generator, self).__init__()

        self.nLatent = nLatent
        self. z_dim = 100

        self.gen = nn.Sequential(
            self.get_generator_block(self.z_dim, nLatent * 4, kernel_size=3, stride=2),
            self.get_generator_block(nLatent * 4, nLatent * 2, kernel_size=4, stride = 1),
            self.get_generator_block(nLatent * 2, nLatent, kernel_size=3, stride = 2),
            self.get_generator_final_block(nLatent, 1, kernel_size=4, stride=2)
        )
        
        
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
        #print(noise.shape)
        #self.gen(noise)
        #x = noise
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        imChannel = 1
        hiddenDimension = 16

        self.disc = nn.Sequential(            
            self.get_critic_block(imChannel, hiddenDimension * 4, kernel_size=4, stride=2),
            self.get_critic_block(hiddenDimension * 4, hiddenDimension * 8, kernel_size=4, stride=2,),
            self.get_critic_final_block(hiddenDimension * 8, 1, kernel_size=4, stride=2,),
        )

    def get_critic_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
                #nn.BatchNorm2d(output_channel),
                nn.GroupNorm(int(output_channel/8),output_channel),
               #  nn.LayerNorm(normalized_shape=self[0,:,:,:].shape),
                nn.LeakyReLU(0.2, inplace=True)
        )
    
    def get_critic_final_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0):
        return  nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            )
    
    def forward(self, image):
        return self.disc(image)




def compute_gp(netD, real_data, fake_data):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        
        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data
        
        # get logits for interpolated images
        interp_logits = netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)