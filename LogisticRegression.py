'''
Name: Logistic Regression Function
Author: Mitchel L. Huott
Date: 07/02/2024
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def LogisticProbability(intercept, beta, X):
    probability = np.exp(intercept + np.dot(beta, X)) / (1 + np.exp(intercept + np.dot(beta, X)))

    return probability

#penalty = -(y*log(p) + (1-y)*log(1-p))
def LogisticLoss(y, probability):
    loss = -(y * np.log(probability) + (1 - y) * np.log(1 - probability))

    return loss

def GradLogistic(prediction, X):
    logGrad = prediction * (1 - prediction) * np.transpose(-X)

    return logGrad

def NewBeta(predGrad, beta, step):
    beta = beta - step * predGrad

    return beta

def Train(X, y, time, step):
    prediction = np.zeros(X.shape[0])
    intercept = 0
    prediction.fill(0.5)
    beta = np.zeros(X.shape[1])

    for i in range(time):
        pred = LogisticProbability(intercept, prediction, X)
        gradient = GradLogistic(prediction, X)
        loss = LogisticLoss(y, pred)
        beta = NewBeta(pred, gradient, step)
        print(f"Iteration {i + 1}, Loss: {loss}")

    return loss, prediction, intercept

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batchSize = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batchSize)
test_dataloader = DataLoader(test_data, batch_size=batchSize)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Flatten the images.
X_flat = X.view(X.size(0), -1).numpy()
y_numpy = y.numpy()

iteration = 10000
learningRate = 0.0001

loss = Train(X_flat, y_numpy, iteration, learningRate)

print(f"Final loss: {loss[0]}")



