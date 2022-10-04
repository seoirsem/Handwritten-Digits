import os
import torch
from torch import nn
from os.path import exists
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils
from models import Generator, Classifier



def main():

 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    nGeneratorIn = 100
    generatorFile = 'models/generator.pt'
    classifierPath = 'models/classifier.pt'

    showImageGrid = True


    classifier = Classifier().to(device)
    if exists(classifierPath):
        checkpoint = torch.load(classifierPath)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded model at ' + classifierPath)
    else:
        raise ValueError('The classifier model file ' + classifierPath + ' was not found.')

    generator = Generator(nGeneratorIn).to(device)
    if exists(generatorFile):
        checkpoint = torch.load(generatorFile, map_location = torch.device(device))
        generator.load_state_dict(checkpoint['model_state_dict'])
        print('Loaded model at ' + generatorFile)
    else:
        raise ValueError('The generator model file ' + generatorFile + ' was not found.')


    fixed_noise = torch.randn(64, nGeneratorIn, 1, 1, device=device)
    images = generator(fixed_noise).detach().cpu()
    
    grid = vutils.make_grid(images, padding=2, normalize=True)
    

    
    imagesUnpacked = torch.split(images,1,dim = 0)
    
    #print(imagesUnpacked[0]*128 + 128)
    for image in imagesUnpacked:

        prediction = classifier.forward(image*128 +128)[0,:]
        argmax = torch.argmax(prediction)
        max = prediction[argmax]
        print('The number looks like ' + str(argmax.item()) + ' with a classifier confidence of ' + str(round(max.item(),2)))



    if showImageGrid:
        plt.figure(figsize = (8,8))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(grid,(1,2,0)))
        plt.show()  






if __name__ == "__main__":
    main()