from matplotlib import pyplot as plt
import torch


from import_data import import_data, prepare_label_array

def run_test_set(model,testFiles):
    nTest = 300
    data, numbers = import_data(testFiles[0],testFiles[1])
    labels = prepare_label_array(numbers)

    testModelOutputs = model.forward(data[0:nTest,:,:,:])
    m,n = testModelOutputs.shape
    nSample = 0
    nTrue = 0
    for i in range(m):
        nSample += 1
        modelOutput = torch.argmax(testModelOutputs[i]).item()
        number = int(numbers[i].item())
        if number == modelOutput:
            outcome = 'Correct'
            nTrue += 1
        else:
            outcome = 'Incorrect'
        # = torch.argmax(labels[i]).item()
        #print('Model prediction was ' + str(modelOutput) + ' against a ground truth of ' + str(number) + '. ' + outcome)

    percentCorrect = round(100.0*nTrue/nSample,1)

    print('In the test set ' + str(percentCorrect) + '% were correctly classified.')



# TODO: output some random subset of incorrectly labelled data for analysis