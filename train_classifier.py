from pydoc import ispath
from random import randint
import numpy as np
from matplotlib import pyplot as plt
import random
import torch
from torch import nn
from torch.utils.data import DataLoader
from os.path import exists
import time

from view_data import plot_single_sample, plot_multiple_samples, plot_multiple_samples_incorrect_label
from import_data import import_data, prepare_label_array
from models import Classifier



def run_backpropogation_optimisation(model,X,y,epochs,initialEpoch,learningRate,printSubset):

    loss_function = nn.BCELoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=learningRate)
    losses = []
    start = time.time()
    epochNumbers = []
    for epoch in range(epochs):
        epochNumbers.append(epoch+initialEpoch)
        yPrediction = model(X.float())
        
        loss = loss_function(yPrediction.reshape(-1).float(), y.reshape(-1).float())
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()
        optimiser.step()
        if epoch % printSubset == 0:
            if epoch != 0:
                print(str(epoch) + ' steps into training the loss is ' + str(round(loss.item(),3)))


    end = time.time()
    print('Total training time: ' + str(round(end - start,2)) + 's for ' + str(epochs) + ' epochs at a learning rate of ' + str(learningRate) + '.')
    
    return model, losses, optimiser, epochNumbers

def run_test_set(model,testFiles,numberToPlot,nTest):
    data, numbers = import_data(testFiles[0],testFiles[1])
    labels = prepare_label_array(numbers)

    testModelOutputs = model.forward(data[0:nTest,:,:,:])
    m,n = testModelOutputs.shape
    nSample = 0
    nTrue = 0
    incorrectData = []
    incorrectModelOutputs = []
    for i in range(m):
        nSample += 1
        modelOutput = torch.argmax(testModelOutputs[i]).item()
        number = int(numbers[i].item())
        if number == modelOutput:
            outcome = 'Correct'
            nTrue += 1
        else:
            outcome = 'Incorrect'
            incorrectData.append([data[i,0,:,:], number, modelOutput])
            incorrectModelOutputs.append(testModelOutputs[i])
        # = torch.argmax(labels[i]).item()
        #print('Model prediction was ' + str(modelOutput) + ' against a ground truth of ' + str(number) + '. ' + outcome)

    percentCorrect = round(100.0*nTrue/nSample,1)

    print('In the test set ' + str(percentCorrect) + '% were correctly classified.')

    if numberToPlot != 0:
        if len(incorrectData) >numberToPlot:
            randPlots = numberToPlot
        else:
            randPlots = len(incorrectData)
        dataSubset = random.sample(incorrectData,randPlots)
        print(randPlots)
        #for i in range(randPlots):
        #    plot_single_sample_incorrect_label(dataSubset[i][0],dataSubset[i][1],dataSubset[i][2])
        plot_multiple_samples_incorrect_label(dataSubset)


def main():

    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    testFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    modelPath = 'models/classifier.pt'
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
    nTrain = 600
    nTest = 10000



    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    data,numbers = import_data(trainingFiles[0],trainingFiles[1])
    labels = prepare_label_array(numbers)

    if plotRandomData:
        n = np.random.randint(0,59999,4)
        plot_multiple_samples(data,numbers,n)

    #either load the model in the file or make a new one
    model = Classifier().to(device)
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