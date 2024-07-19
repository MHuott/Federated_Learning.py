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


def LogisticProbability(beta, X, intercept):
    y = np.dot(X, beta) + intercept

    if y.ndim >= 1:
        const = np.ones(X.shape) * -2
        X = np.maximum(X, const)
    else:
        X = np.max(X,-2)

    probability = 1 / (1 + np.exp(-y))
    #print(probability)

    return probability


#penalty = -(y*log(p) + (1-y)*log(1-p))
def LogisticLoss(y, probability):
    epsilon = 1e-15

    probability =  np.clip(probability,epsilon, 1 - epsilon)
    #print("probability: ", probability)
    #print("y: ", y)
    loss = -(np.dot(y.T, np.log(probability)) + np.dot((1 - y), np.log(1 - probability)))
    return loss.mean()  # Return the mean loss

#intercept gradient is average difference between ground truth and probability
def GradLogistic(probability, X, y):
    betaGrad = np.dot((probability - y), X)
    inteGrad = probability - y

    return betaGrad, inteGrad


def NewBeta(beta, gradient, step):
    beta = beta - np.dot(step, gradient)

    return beta

def NewIntercept(intercept, gradient, step):
    intercept = intercept - np.dot(step, gradient)

    return intercept

def Train(X, y, time, step):
    prediction = np.zeros(X.shape[0])
    beta = np.ones(X.shape[1])
    beta = beta * -0.00001
    intercept = np.ones(X.shape[0])
    loss = 0

    for i in range(time):
        probability = LogisticProbability(beta, X, intercept)
        loss = LogisticLoss(y, probability)
        gradient = GradLogistic(probability, X, y)
        beta = NewBeta(beta, gradient[0], step)
        intercept = NewIntercept(intercept, gradient[1], step)
        print(f"Iteration {i + 1}, Loss: {loss}")

    return loss, beta, intercept

data = pd.read_csv("C:/Users/mlhuo_dkvynem/OneDrive/Desktop/Raisin_Dataset.csv", sep=',')
data['Class']= pd.factorize(data['Class'])[0]

X = data.drop('Class', axis=1)
y = data['Class']

Train(X=X, y=y, time=20, step=0.0001)
