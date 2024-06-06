'''
Name: Basic Machine Learning Algorithm
Author: Mitchel L. Huott
Date: 06/04/2024
'''

import numpy as np
import math
import random as ran

def Prediction(beta, beta0, X):
    z = np.dot(beta, X)
    z = np.add(z, beta0)

    return z

def Loss(y, fun):
    loss = np.sum(np.square(y - fun))

    return loss

def Gradient():


    return grad




matrixSize = 4
pred = np.zeros((matrixSize, 1))
grad = np.zeros((matrixSize, 1))


print(pred)

X = np.ones((matrixSize,matrixSize))

for i in range(matrixSize):
    for j in range(matrixSize):
        X[i][j] = X[i][j] * ran.randint(0,10)


outputSize = matrixSize
y = np.ones(outputSize)

for i in range(outputSize):
    y[i] = y[i] * ran.randint(1,5)

print("Matrix")
print(X)
print("Output")
print(y)


bias = 1
beta = np.ones((1,matrixSize))

print("Beta")
print(beta)

for i in range(matrixSize):
    pred[i] = Prediction(beta, bias, X[i,:])

'''print("Prediction Values")
for i in range(matrixSize):
    print(pred[i])
    break'''

L = Loss(y, pred)

print("Loss")
print(L)

#gradient descent loser


