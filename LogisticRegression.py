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
from datasets import load_dataset
import pandas as pd


def LogisticProbability(intercept, beta, X):
    probability = np.exp(intercept + np.dot(X, beta)) / (1 + np.exp(intercept + np.dot(X, beta)))

    return probability


#penalty = -(y*log(p) + (1-y)*log(1-p))
def LogisticLoss(y, probability):
    loss = -(np.dot(y.T, np.log(probability)) + np.dot((1 - y).T, np.log(1 - probability)))

    return loss


def GradLogistic(prediction, X):
    logGrad = prediction * (1 - prediction) * (-1 * np.transpose(X))

    return logGrad


def NewBeta(predGrad, beta, step):
    beta = beta - step * predGrad

    return beta


def Train(X, y, time, step):
    prediction = np.zeros(X.shape[0])
    intercept = 0
    beta = np.ones(X.shape[1])
    print("Beta")
    print(beta.shape)


    for i in range(time):
        probability = LogisticProbability(intercept, beta, X)
        print("Probability")
        print(probability.shape)
        loss = LogisticLoss(y, probability)
        print(probability.shape)
        gradient = GradLogistic(prediction, X)
        beta = NewBeta(probability, gradient, step)
        print(f"Iteration {i + 1}, Loss: {loss}")

    return loss, beta, intercept

data = pd.read_csv("C:/Users/mlhuo_dkvynem/OneDrive/Desktop/Raisin_Dataset.csv", sep=',')
data['Class']= pd.factorize(data['Class'])[0]

print(data)

X = data.drop('Class', axis=1)
y = data['Class']

print("y")
print(y.shape)
print("X")
print(X.shape)
Train(X=X, y=y, time=10, step=0.001)
