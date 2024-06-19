'''
Name: Basic Machine Learning Algorithm
Author: Mitchel L. Huott
Date: 06/04/2024
'''

import numpy as np
import math
import random as ran
import matplotlib.pyplot as plt

#f(x) = B^TX + B0
def Prediction(beta, beta0, X):
    z = np.dot(beta, X) + beta0

    return z

#L = (y - f(x))^2
def Loss(y, fun):
    loss = np.sum(np.square(y - fun))

    return loss

def Gradient(X, y, pred):
    gradL = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        gradL[j] = -2 * np.sum((y - pred) * X[:,j])

    return gradL

def NewBeta(gradL, beta):
    step = 0.001
    beta = beta - step * gradL

    return beta

def NewBias(y, bias, pred):
    bias = bias - 0.001 * -2 * np.sum(y - pred)

    return bias

def Train(X,y,time):
    bias = 1
    beta = np.zeros(X.shape[1])
    pred = Prediction(beta, bias, X)

    L = Loss(y, pred)


    for i in range(time):
        newPred = Prediction(beta, bias, X)
        gradient = Gradient(X, y, newPred)
        beta = NewBeta(gradient, beta)
        bias = NewBias(y, bias, newPred)
        L = Loss(y, newPred)
        print(f"Iteration {i + 1}, Loss: {L}")

    return L, beta, bias

def Application(beta, bias, X):
    z = 0
    for i in range(len(X)):
        z += beta[i] * X[i]
    z += bias

    return z


matrixSize = 4

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
print(y)
iteration = 100

loss = Train(X, y, iteration)

print(f"Final loss: {loss[0]}")

print("Beta")
print(loss[1])

print("Bias")
print(loss[2])

a = [2,3,4,5]
print("Guess")
print(Application(loss[1], loss[2], a))






