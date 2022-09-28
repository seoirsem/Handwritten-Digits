import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import torch.optim as optim
from matplotlib import pyplot as plt


from import_data import generate_random_image_data, import_data, prepare_label_array
from view_data import plot_single_sample

class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        ngf = 32
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( 1, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)

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



def main():

    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    testFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    digitClassifierModelPath = 'savedModel.pt'

    realData,numbers = import_data(trainingFiles[0],trainingFiles[1])
    realDataLabels = prepare_label_array(numbers)

    epochs = 100
    printSubset = 10
    learningRate = 0.0002
    nTrain = 100
    plotLearningRate = True

    nc = 1
    nz = 100
    # Size of feature maps in generator
    ngf = 28
    # Size of feature maps in discriminator
    ndf = 28
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)


    generatedImage = generate_random_image_data(2)
    #plot_single_sample(generatedImage[0,0,:,:],'Generated')
    # output = discriminator.forward(generatedImage)
    # print(output.shape)
    # print(output)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, 1, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for each network
    optimiserD = optim.Adam(discriminator.parameters(), lr=learningRate, betas=(beta1, 0.999))
    optimiserG = optim.Adam(generator.parameters(), lr=learningRate, betas=(beta1, 0.999))
    
    losses = []
    X = realData[0:nTrain,:,:,:].float()
    labelReal = torch.full((X.shape[0],), real_label, dtype=torch.float, device=device)
    Xfake = generate_random_image_data(nTrain)
    labelFake = torch.full((Xfake.shape[0],), fake_label, dtype=torch.float, device=device)
    for epoch in range(epochs):
        
        yPrediction = discriminator(X)
        #print(yPrediction)
        loss = criterion(yPrediction.reshape(-1).float(), labelReal.reshape(-1).float())
        
        yPrediction = discriminator(Xfake)
        loss = criterion(yPrediction.reshape(-1).float(), labelFake.reshape(-1).float())

        losses.append(loss.item())
        discriminator.zero_grad()
        loss.backward()
        optimiserD.step()
        if epoch % printSubset == 0:
            if epoch != 0:
                print(str(epoch) + ' steps into training the loss is ' + str(round(loss.item(),3)))

    if plotLearningRate:
        plt.figure()
        plt.plot(losses)
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.show()

if __name__ == "__main__":
    main()