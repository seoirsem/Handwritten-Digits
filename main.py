from pydoc import ispath
from random import randint
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from os.path import exists
import math

from view_data import plot_single_sample, plot_multiple_samples
from import_data import import_data, prepare_label_array
from neural_network import NeuralNetwork, run_backpropogation_optimisation
from test_network import run_test_set

def main():

    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    testFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    modelPath = 'savedModel.pt'
    # header    
    loadModel = True
    saveModel = True
    plotRandomData = False
    plotLearningRate = False
    # set 0 to not plot any. Otherwise plot n incorrectly labelled items
    plotNumberIncorrectSubset = 6
    epochs = 1
    printSubsets = 10 # how often you output model progress
    learningRate = 0.1
    nTrain = 60000
    nTest = 10000



    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    data,numbers = import_data(trainingFiles[0],trainingFiles[1])
    labels = prepare_label_array(numbers)

    if plotRandomData:
        n = np.random.randint(0,59999,4)
        plot_multiple_samples(data,numbers,n)

    #either load the model in the file or make a new one
    model = NeuralNetwork().to(device)
    optimiser = torch.optim.SGD(model.parameters(), lr=learningRate)

    if loadModel and exists(modelPath):
        checkpoint = torch.load(modelPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        initialEpoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Loaded model at ' + modelPath)
    else:
        initialEpoch = 0
   
   # print(model.forward(data[0,:,:].reshape(1,1,28,28)))
    
    # TODO split into subsets of n to output progress 
    
    model, losses, optimiser, epochNumbers = run_backpropogation_optimisation(model,data[0:nTrain,:,:,:],labels[0:nTrain,:],epochs,initialEpoch,learningRate,printSubsets)
            
    print('Final loss value of: ' + str(round(losses[-1],3)))

    if saveModel:
        torch.save({
            'epoch': epochNumbers[-1],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimiser.state_dict(),
            'loss': losses[-1],
                }, modelPath)
        print('Model saved as "' + modelPath + '"')


    if plotLearningRate:
        plt.figure()
        plt.plot(epochNumbers,losses)
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.show()

    run_test_set(model,testFiles,plotNumberIncorrectSubset,nTest)


if __name__ == "__main__":
    main()