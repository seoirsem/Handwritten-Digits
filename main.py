from random import randint
import numpy as np
from matplotlib import pyplot as plt
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from view_data import plot_single_sample, plot_multiple_samples
from import_data import import_data, prepare_label_array
from neural_network import NeuralNetwork, run_backpropogation_optimisation
from test_network import run_test_set

def main():

    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    testFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    modelPath = ''
    # header    
    plotRandomData = False
    plotLearningRate = True
    epochs = 100
    learningRate = 0.1
    nTrain = 1000


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    data,numbers = import_data(trainingFiles[0],trainingFiles[1])
    labels = prepare_label_array(numbers)

    if plotRandomData:
        n = np.random.randint(0,59999,4)
        plot_multiple_samples(data,numbers,n)


    model = NeuralNetwork().to(device)   
   # print(model.forward(data[0,:,:].reshape(1,1,28,28)))
    
    # TODO split into subsets of n to output progress 
    model, losses = run_backpropogation_optimisation(model,data[0:nTrain,:,:,:],labels[0:nTrain,:],epochs,learningRate)

    # TODO: save intermediate training steps
    # https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch

    print('Final loss value of: ' + str(round(losses[-1],2)))

    if plotLearningRate:
        plt.figure()
        plt.plot(losses)
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Binary Cross Entropy Loss')
        plt.show()

    run_test_set(model,testFiles)


if __name__ == "__main__":
    main()