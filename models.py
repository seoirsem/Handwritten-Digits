import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
from torch import autograd
from os.path import exists


class Generator(nn.Module):
    
    def __init__(self,nLatent):
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
        x = noise.view(len(noise), self.z_dim, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.convFinal(x)
        return x
    
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

    def get_critic_block(self, input_channel, output_channel, kernel_size, stride = 1, padding = 2):
        return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
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




class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = self.conv_block(1,10,3)
        self.conv2 = self.conv_block(10,20,2,1,2)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def conv_block(self, inputChannel, outputChannel, kernelSize, stride = 1, padding = 1):
      return nn.Sequential(
          nn.Conv2d(inputChannel,outputChannel,kernelSize,stride,padding),
          nn.MaxPool2d(3),
          nn.BatchNorm2d(outputChannel),
          nn.ReLU(0.2)
      )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        return x



def return_runtime_model_versions():
    ############### this script loads the training models and saves just the last model version ##########
    ######### the training models are large and save previous model iterations #############

    generatorFile = 'models/generator.pt'
    discriminatorFile = 'models/discriminator.pt'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    discriminator = Discriminator().to(device)
    if exists(discriminatorFile):
        checkpoint = torch.load(discriminatorFile, map_location = torch.device(device))
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded model at ' + discriminatorFile)

    generator = Generator(100).to(device)
    if exists(generatorFile):
        checkpoint = torch.load(generatorFile, map_location = torch.device(device))
        generator.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded model at ' + generatorFile)

    generatorRuntime = 'models/generatorRuntime.pt'
    discriminatorRuntime = 'models/discriminatorRuntime.pt'

    torch.save(generator, generatorRuntime)
    torch.save(discriminator, discriminatorRuntime)

    ## to load, simply: ###########
    # model = torch.load(PATH)
    # model.eval()

    # Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results.
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html

    print('Files saved')