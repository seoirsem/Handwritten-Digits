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
from torch.autograd import Variable


from import_data import generate_random_image_data, import_data, prepare_label_array
from view_data import plot_single_sample

from GAN_models import Discriminator, Generator, compute_gp

def get_random_subset(imageData,n):
    indices = np.random.randint(0,imageData.shape[0],n)
    print(indices)
    return imageData[indices,:,:,:]

def decorate_with_diffs(data, exponent):
    mean = torch.mean(data.data, 1)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    return torch.cat([data, diffs], 1)


def main():

    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    testFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    digitClassifierModelPath = 'savedModel.pt'

    generatorFile = 'generator_small.pt'
    discriminatorFile = 'discriminator_small.pt'
    loadModel = True
    saveModel = True
    plotGeneratedImages = True
    plotSingleSample = True

    realData,numbers = import_data(trainingFiles[0],trainingFiles[1])
    epochs = 20
    MNISTSubset = 1280#6400#int(12800/2)

    # https://arxiv.org/pdf/1701.07875.pdf Wasserstein paper for source of values
    learningRate = 0.00005
    #learningRate = 0.0005
    clipping = 0.02
    nBatch = 128
    nCritic = 3 # number of generations of the critic per generator iteration
    lamdaGP = 10 # https://arxiv.org/pdf/1704.00028.pdf
    plotLearningRate = True
    workers = 2

    nGeneratorIn = 100


    # ### Uncomment only one of these
    #(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
 
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

    generator = Generator(nGeneratorIn).to(device)
    if loadModel and exists(generatorFile):
        checkpoint = torch.load(generatorFile)
        generator.load_state_dict(checkpoint['model_state_dict'])
        #optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        #initialEpoch = checkpoint['epoch']
        #loss = checkpoint['loss']
        print('Loaded model at ' + generatorFile)
    else:
        initialEpoch = 0
        
    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    dataloader = torch.utils.data.DataLoader(realData[0:MNISTSubset,:,:,:], batch_size=nBatch, shuffle=True, num_workers=workers)

    # Setup Adam optimizers for each network
    optimiserD = optim.Adam(discriminator.parameters(), lr=learningRate)
    optimiserG = optim.Adam(generator.parameters(), lr=learningRate)
    
    lossesGenerator = []
    lossesDiscriminator = []
    imageList = []
    fixed_noise = torch.randn(64, nGeneratorIn, 1, 1, device=device)
    steps = []
    step = 0


    for epoch in range(epochs):
        epochStartTime = time.time()
        for i, data in enumerate(dataloader, 0):
            #print(i)
            steps.append(step)
            step += 1
            for j in range(nCritic):
            ############ Train discriminator #############
                optimiserD.zero_grad()

                # all real
                discriminatorPredictionReal = discriminator(data)
                meanPredReal = torch.mean(discriminatorPredictionReal)
                # all fake
                noise = torch.randn(nBatch, nGeneratorIn, 1, 1, device=device)
                generatedDigits = generator.forward(noise)
                discriminatorPredictionFake = discriminator(generatedDigits)
                meanPredFake = torch.mean(discriminatorPredictionFake)
                # calculate loss. This gets higher the better the discriminator is
                discriminatorLoss = - meanPredReal + meanPredFake + lamdaGP * compute_gp(discriminator,data,generatedDigits)
                discriminatorLoss.backward()

                
                # step
                optimiserD.step()
                #for par in discriminator.parameters():
                 #   par.data.clamp_(-clipping,clipping)

            ############ Train Generator ##############
            optimiserG.zero_grad()
            # Since we just updated discriminator, perform another forward pass of all-fake batch through D
            output = discriminator(generator(noise))
            # Calculate generator loss based on this output
            generatorError = -torch.mean(output)
            # Calculate gradients for generator
            generatorError.backward()            
            # Update generator
            optimiserG.step()
            meanYGenerator = output.mean().item()

            lossesDiscriminator.append(discriminatorLoss.item())
            lossesGenerator.append(generatorError.item())
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     discriminatorLoss.item(), generatorError.item(), meanPredReal, meanPredFake, meanYGenerator))
        
            if (step % 500 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                    imageList.append(vutils.make_grid(fake, padding=2, normalize=True))
        delT = time.time() - epochStartTime
        print("Epoch took " + str(round(delT)) + "s.")
        ### TODO ### output images of generator during training
    if plotGeneratedImages:
        plt.figure(figsize = (8,8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(imageList[-1],(1,2,0)))
        plt.show()  
    if plotSingleSample:  
        plot_single_sample(generator.forward(torch.randn(1, nGeneratorIn, 1, 1, device=device)).detach()[0,0,:,:],"Generated Sample")

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
        maxD = abs(max(lossesDiscriminator,key = abs))
        plt.plot([x/maxD for x in lossesDiscriminator], label = 'Discriminator')
        maxG = abs(max(lossesGenerator,key = abs))
        plt.plot([x/maxG for x in lossesGenerator], label = 'Generator')
        plt.legend()
        plt.grid()
        plt.xlabel('Model Steps')
        plt.ylabel('Loss (Normalised)')
        plt.show()

if __name__ == "__main__":
    main()