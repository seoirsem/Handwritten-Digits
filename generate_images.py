import os
import torch
from torch import nn
from os.path import exists
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils
from models import Generator, Classifier
import math

def show_titled_grid(images,titles,ncol):
    h = math.floor(images.shape[0]/ncol)
    if ncol != 1:
        fig, ax = plt.subplots(h, ncol)
        for i in range(images.shape[0]):
            x = math.floor(i/ncol)
            y = i % ncol    
            ax[x, y].imshow(images[i,0,:,:],cmap='gray')
            ax[x,y].axis('off')
            ax[x,y].set_title('Pr: %i \nCon: %5.1f' % (titles[i][0], titles[i][1]))
    else:   
        fig, ax = plt.subplots(1, h)
        for i in range(images.shape[0]):
            x = math.floor(i/ncol)
            ax[x].imshow(images[i,0,:,:],cmap='gray')
            ax[x].axis('off')
            ax[x].set_title('Pr: %i \nCon: %5.1f' % (titles[i][0], titles[i][1]))
    fig.tight_layout()
    plt.show()

def main():

 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    nGeneratorIn = 100
    generatorFile = 'models/generator.pt'
    classifierPath = 'models/classifier.pt'

    showSimpleImageGrid = False
    printPredictions = False

    classifier = Classifier().to(device)
    if exists(classifierPath):
        checkpoint = torch.load(classifierPath, map_location=torch.device(device))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded model at ' + classifierPath)
    else:
        raise ValueError('The classifier model file ' + classifierPath + ' was not found.')

    generator = Generator(nGeneratorIn).to(device)
    if exists(generatorFile):
        generator = torch.load(generatorFile, map_location = torch.device(device))
        generator.eval()
        print('Loaded model at ' + generatorFile)
    else:
        raise ValueError('The generator model file ' + generatorFile + ' was not found.')


    fixed_noise = torch.randn(100, nGeneratorIn, 1, 1, device=device)
    images = generator(fixed_noise).detach().cpu()
    
    imagesUnpacked = torch.split(images,1,dim = 0)
    
    labels = []
    classes = []
    for image in imagesUnpacked:

        prediction = classifier.forward(image*128 +128)[0,:]
        argmax = torch.argmax(prediction)
        max = prediction[argmax]
        labels.append([argmax.item(),round(max.item(),2)])
        classes.append(argmax.item())
        #print('The number looks like ' + str(argmax.item()) + ' with a classifier confidence of ' + str(round(max.item(),2)))


    # Here we calculate the distribution of the numbers of generated digits
    classDist = [0]*10
    for c in classes:
        classDist[c] += 1

    print(classDist)
    plt.figure()
    plt.grid(zorder=0)
    plt.hist(classes,density = True,zorder = 3)
    plt.ylabel('Proportion')
    plt.xlabel('Generated Digit')
    plt.show()

    if printPredictions:
        show_titled_grid(images,labels,1)


    if showSimpleImageGrid:
        grid = vutils.make_grid(images, padding=2, normalize=True)
        plt.figure(figsize = (8,8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(grid,(1,2,0)))
        plt.show()  






if __name__ == "__main__":
    main()