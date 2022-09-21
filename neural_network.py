import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.sigmoid(x)


def run_backpropogation_optimisation(model,X,y,epochs,learningRate):

    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    losses = []
    start = time.time()
    for epoch in range(epochs):

        yPrediction = model(X.float())
        
        loss = loss_function(yPrediction.reshape(-1).float(), y.reshape(-1).float())
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.time()
    print('Total training time: ' + str(round(end - start,2)) + 's for ' + str(epochs) + ' epochs at a learning rate of ' + str(learningRate) + '.')
    
    return model, losses