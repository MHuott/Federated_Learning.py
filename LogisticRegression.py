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

def LogisticPrediction(intercept, beta, X):
    prediction = np.exp(intercept + np.dot(beta, X)) / (1 + np.exp(intercept + np.dot(beta, X)))

    return prediction

#penalty = -(y*log(p) + (1-y)*log(1-p))
def LogisticLoss(y, p):
    loss = -(y * np.log(y) + (1 - y) * np.log(1 - y))

    return loss

def GradLogistic():

    return

