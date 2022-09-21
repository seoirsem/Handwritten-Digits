import idx2numpy
import numpy as np
import torch

def import_data(imageFile,labelsFile):
    data = idx2numpy.convert_from_file(imageFile)
    # arr is a np.ndarray type of object of shape (60000, 28, 28)
    labels = idx2numpy.convert_from_file(labelsFile)
    # labels is a np.ndarray type of object of shape (60000, )
    data = torch.tensor(data).float()
    labels = torch.tensor(labels).float()
    dataShape = torch.reshape(data,(-1,1,28,28))
    return dataShape, labels
    

def prepare_label_array(labels):
    # input is an array of the digits, output is an array of 10x1 with a "1" at the correct position
    m = list(labels.shape)[0]

    labelTensor = torch.zeros(size=(m,10))
    for i in range(m):
        l = int(labels[i].item())
        labelTensor[i,l] = 1

    return labelTensor