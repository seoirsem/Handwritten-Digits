from random import randint
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from view_data import plot_single_sample, plot_multiple_samples
from import_data import import_training_data
from neural_network import NeuralNetwork

def main():
    
    printRandomData = False
    



    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    data,labels = import_training_data()
    if printRandomData:
        n = np.random.randint(0,59999,4)
        plot_multiple_samples(data,labels,n)

    model = NeuralNetwork().to(device)
    #print(model)
    

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    print(pred_probab)


if __name__ == "__main__":
    main()