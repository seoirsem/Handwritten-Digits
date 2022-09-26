from matplotlib import pyplot as plt
import torch
import random

from import_data import import_data, prepare_label_array
from view_data import plot_single_sample_incorrect_label, plot_multiple_samples_incorrect_label

def run_test_set(model,testFiles,numberToPlot,nTest):
    data, numbers = import_data(testFiles[0],testFiles[1])
    labels = prepare_label_array(numbers)

    testModelOutputs = model.forward(data[0:nTest,:,:,:])
    m,n = testModelOutputs.shape
    nSample = 0
    nTrue = 0
    incorrectData = []
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



# TODO: output some random subset of incorrectly labelled data for analysis

