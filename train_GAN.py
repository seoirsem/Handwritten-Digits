import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import torch.optim as optim
from matplotlib import pyplot as plt
import numpy as np
from os.path import exists
import torchvision.utils as vutils


from import_data import generate_random_image_data, import_data, prepare_label_array
from view_data import plot_single_sample

from GAN_models import Discriminator, Generator

def get_random_subset(imageData,n):
    indices = np.random.randint(0,imageData.shape[0],n)
    print(indices)
    return imageData[indices,:,:,:]



def main():

    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    testFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    digitClassifierModelPath = 'savedModel.pt'

    generatorFile = 'generator.pt'
    discriminatorFile = 'discriminator.pt'
    loadModel = True
    saveModel = True
    plotGeneratedImages = True

    realData,numbers = import_data(trainingFiles[0],trainingFiles[1])
    realDataLabels = prepare_label_array(numbers)
    epochs = 1
    printSubset = 10
    learningRate = 0.0002
    plotLearningRate = True
    nBatch = 128
    workers = 2

    nc = 1
    nGeneratorIn = 100
    # Size of feature maps in generator
    ngf = 28
    # Size of feature maps in discriminator
    ndf = 28
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    discriminator = Discriminator().to(device)
    if loadModel and exists(discriminatorFile):
        checkpoint = torch.load(discriminatorFile)
        discriminator.load_state_dict(checkpoint['model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        #initialEpoch = checkpoint['epoch']
        #loss = checkpoint['loss']
        print('Loaded model at ' + discriminatorFile)
    else:
        initialEpoch = 0

    generator = Generator().to(device)
    if loadModel and exists(generatorFile):
        checkpoint = torch.load(generatorFile)
        generator.load_state_dict(checkpoint['model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        #initialEpoch = checkpoint['epoch']
        #loss = checkpoint['loss']
        print('Loaded model at ' + generatorFile)
    else:
        initialEpoch = 0
        
    #generator.forward(noise)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator

    # Establish convention for real and fake labels during training
    labelReal = 1.
    labelFake = 0.

    dataloader = torch.utils.data.DataLoader(realData[0:12800,:,:,:], batch_size=nBatch, shuffle=True, num_workers=workers)

    # Setup Adam optimizers for each network
    optimiserD = optim.Adam(discriminator.parameters(), lr=learningRate, betas=(beta1, 0.999))
    optimiserG = optim.Adam(generator.parameters(), lr=learningRate, betas=(beta1, 0.999))
    
    lossesGenerator = []
    lossesDiscriminator = []
    imageList = []
    fixed_noise = torch.randn(64, nGeneratorIn, 1, 1, device=device)
    steps = []
    step = 0

    for epoch in range(epochs):
        
        for i, data in enumerate(dataloader, 0):
            #print(i)
            steps.append(step)
            step += 1
        ############ Train discriminator #############
            discriminator.zero_grad()
            # all real
            label = torch.full((nBatch,), labelReal, dtype=torch.float, device=device)
            yPrediction = discriminator(data)
            errorReal = criterion(yPrediction.reshape(-1).float(), label.reshape(-1).float())   
            errorReal.backward()
            meanYReal = yPrediction.mean().item()

            # all fake
            noise = torch.randn(nBatch, nGeneratorIn, 1, 1, device=device)
            xFake = generator.forward(noise)
            #xFake = generate_random_image_data(nBatch) ## replace with generator output
            label.fill_(labelFake)
            yPrediction = discriminator(xFake)
            errorFake = criterion(yPrediction.reshape(-1).float(), label.reshape(-1).float())
            errorFake.backward()
            meanYFake = yPrediction.mean().item()
            
            # step
            discriminatorError = errorReal + errorFake
            optimiserD.step()

        ############ Train Generator ##############
            generator.zero_grad()
            label.fill_(labelReal)
            # fake labels are real for generator cost
            # Since we just updated discriminator, perform another forward pass of all-fake batch through D
            output = discriminator(xFake.detach()).view(-1)
            # Calculate generator loss based on this output
            generatorError = criterion(output, label.reshape(-1).float())
            # Calculate gradients for generator
            generatorError.backward()            
            # Update generator
            optimiserG.step()
            meanYGenerator = output.mean().item()

            lossesDiscriminator.append(discriminatorError.item())
            lossesGenerator.append(generatorError.item())
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     discriminatorError.item(), generatorError.item(), meanYReal, meanYFake, meanYGenerator))
        
            if (step % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                    imageList.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        ### TODO ### output images of generator during training
    if plotGeneratedImages:
        plt.figure(figsize = (8,8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(imageList[-1],(1,2,0)))
        plt.show()

    if saveModel:
        torch.save({
            #'epoch': epochs[-1],
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimiserG.state_dict(),
            'loss': lossesGenerator[-1],
                }, generatorFile)
        print('Model saved as "' + generatorFile + '"')

        torch.save({
            #'epoch': epochs[-1],
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimiserD.state_dict(),
            'loss': lossesDiscriminator[-1],
                }, discriminatorFile)
        print('Model saved as "' + discriminatorFile + '"')




    if plotLearningRate:
        plt.figure()
        plt.plot(lossesDiscriminator, label = 'Discriminator')
        plt.plot(lossesGenerator, label = 'Generator')
        plt.legend()
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.show()

if __name__ == "__main__":
    main()