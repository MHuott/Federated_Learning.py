'''
Name: Basic Machine Learning Algorithm
Author: Mitchel L. Huott
Date: 06/04/2024
'''

import numpy as np
import math
import random as ran
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# Function to compute prediction
#f(x) = B^TX + B0
def Prediction(beta, beta0, X):
    z = np.dot(X, beta) + beta0

    return z


# Function to compute loss
'''
We divide by the len of y to normalize the loss.  
len(y) is the number of samples
'''
def LinearLoss(y, fun):
    loss = np.sum(np.square(y - fun)) / len(y)

    return loss


# Function to compute gradient
def Gradient(X, y, pred):
    gradL = -2 * np.dot(X.T, (y - pred)) / len(y)

    return gradL


'''def StochGradient(X, y, pred, batchSize = 1):
    index = np.random.randint(0, X.shape[0], 1)'''

'''def SGD(X, y, lr=0.05, epoch=10, batch_size=1):
    
    #Stochastic Gradient Descent for a single feature
    

    m, b = 0.5, 0.5  # initial parameters
    log, mse = [], []  # lists to store learning process

    for _ in range(epoch):
        indexes = np.random.randint(0, len(X), batch_size)  # random sample

        Xs = np.take(X, indexes)
        ys = np.take(y, indexes)
        N = len(Xs)

        f = ys - (m * Xs + b)

        # Updating parameters m and b
        m -= lr * (-2 * Xs.dot(f).sum() / N)
        b -= lr * (-2 * f.sum() / N)

        log.append((m, b))
        mse.append(mean_squared_error(y, m * X + b))

    return m, b, log, mse'''


# Function to update beta
def NewBeta(gradL, beta, step):
    beta = beta - step * gradL

    return beta


# Function to update bias
def NewBias(y, bias, pred, step):
    bias = bias - step * -2 * np.sum(y - pred) / len(y)

    return bias


# Function to update bias
def Train(X, y, time, step):
    bias = 1
    beta = np.zeros(X.shape[1])

    for i in range(time):
        pred = Prediction(beta, bias, X)
        gradient = Gradient(X, y, pred)
        #gradient = SGD(X, y, lr=0.01, epoch=10, batch_size=1)
        beta = NewBeta(gradient, beta, step)
        bias = NewBias(y, bias, pred, step)
        L = LinearLoss(y, pred)
        print(f"Iteration {i + 1}, Loss: {L}")

    return L, beta, bias


# Function to apply the model to a new input
def Application(beta, bias, X):
    z = np.dot(X, beta) + bias

    return z


'''matrixSize = 4

X = np.ones((matrixSize,matrixSize))

for i in range(matrixSize):
    for j in range(matrixSize):
        X[i][j] = ran.randint(0,10)

y = np.ones(matrixSize)

for i in range(y.shape[0]):
    y[i] = ran.randint(1,5)

print("Matrix")
print(X)
print("Output")
print(y)'''

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

XFlat = X.view(X.size(0), -1).numpy()
yNumpy = y.numpy()

iteration = 10000
learningRate = 0.0001

loss = Train(XFlat, yNumpy, iteration, learningRate)

print(f"Final loss: {loss[0]}")

'''
print("Beta")
print(loss[1])

print("Bias")
print(loss[2])

a = [2, 3, 4, 5]
print("Guess")
print(Application(loss[1], loss[2], a))
'''