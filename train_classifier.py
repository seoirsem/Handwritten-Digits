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


############# Backprop #############

def run_backpropogation_optimisation(device,model,X,y,epochs,learningRate,iterNumbers,
                                     printSubset,losses,nBatch,loss_function,dataloader,percentCorrect,plots):
    numberBatch = math.floor(X.shape[0]/nBatch)
    start = time.time()
    n = iterNumbers[-1]
    for epoch in range(epochs):
      epochStart = time.time()
      for i, data in enumerate(dataloader, 0):
        n = n+1
        iterNumbers.append(n)
        optimiser.zero_grad()
        yPrediction = model(data[0])
        loss = loss_function(yPrediction, data[1])
        loss.backward()
        optimiser.step()
        losses.append(loss.item())

        #if i % 20 == 0:
          # This is optional to see how decreasing error feeds into correct classification
         # percentCorrect.append([n,run_test_set(model,testFiles,plotNumberIncorrectSubset,nTest,device,False)])

      end = time.time()
      print('Epoch [' + str(epoch+1) + '/' + str(epochs) + '] training time: ' + str(round(end - start,2)) + 
            's at a learning rate of ' + str(learningRate) + ' with a final loss of ' + str(round(loss.item(),3)))
      if epoch % 10 == 0 and plots:
        percentCorrect.append([n,run_test_set(model,testFiles,plotNumberIncorrectSubset,nTest,device,True)])
    return model, losses, optimiser, iterNumbers,percentCorrect

############ Test model ############

def run_test_set(model,testFiles,numberToPlot,nTest,device,printOutput):
    data, numbers = import_data(testFiles[0],testFiles[1])
    labels = prepare_label_array(numbers)
    data = data.to(device)
    numbers = numbers.to(device)
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
            incorrectData.append([data[i,0,:,:].cpu(), number, modelOutput])
            incorrectModelOutputs.append(testModelOutputs[i])
 
    percentCorrect = round(100.0*nTrue/nSample,1)

    if printOutput:
      print('In the test set ' + str(percentCorrect) + '% were correctly classified.')

    if numberToPlot != 0 and printOutput:
        if len(incorrectData) >numberToPlot:
            randPlots = numberToPlot
        else:
            randPlots = len(incorrectData)
        dataSubset = random.sample(incorrectData,randPlots)
        plot_multiple_samples_incorrect_label(dataSubset)
    return percentCorrect



def load_model(model,optimiser,modelPath):

  if exists(modelPath):
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
    initialEpoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print('Loaded model at ' + modelPath)
  else:
    print('No file found')

  return model,optimiser

def main():

    trainingFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    testFiles = ['data/train-images-idx3-ubyte','data/train-labels-idx1-ubyte']
    modelPath = 'models/classifier.pt'
    # header    

    plotRandomData = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data,numbers = import_data(trainingFiles[0],trainingFiles[1])
    labels = prepare_label_array(numbers)
    data = data.to(device)
    labels = labels.to(device)
    if plotRandomData:
        n = np.random.randint(0,59999,4)
        plot_multiple_samples(data,numbers,n)

    #either load the model in the file or make a new one
    model = Classifier().to(device)

    loss_function = nn.CrossEntropyLoss()
    losses = []
    iterNumbers = [0]
    percentCorrect = []
    dataset = TensorDataset( Tensor(data), Tensor(labels) )


    loadModel = False
    saveModel = True
    # set 0 to not plot any. Otherwise plot n incorrectly labelled items
    plotNumberIncorrectSubset = 6
    epochs = 100
    printSubsets = 10 # how often you output model progress
    learningRate = 0.0005
    nTrain = 60000
    nTest = 10000
    nBatch = 128

    optimiser = torch.optim.Adam(model.parameters(), lr=learningRate)
    dataloader = DataLoader(dataset, batch_size= nBatch)


    if loadModel and exists(modelPath):
        checkpoint = torch.load(modelPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        initialEpoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print('Loaded model at ' + modelPath)
    else:
        initialEpoch = 0

    
    model, losses, optimiser, iterNumbers, percentCorrect = run_backpropogation_optimisation(device,model,data[0:nTrain,:,:,:],labels[0:nTrain,:],epochs,learningRate,
                                                                                            iterNumbers,printSubsets,losses,nBatch,loss_function,dataloader,percentCorrect,True)
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




