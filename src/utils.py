import cooler

import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import datetime


def train_epoch(model, opt, loss, data_iterator, regression=False):
    model.train(True)
    for i in data_iterator:
        x, y = i
        x = x[:, np.newaxis]
        
        if regression:
            X, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        else:
            X, y = torch.from_numpy(x).float(), torch.from_numpy(y)
        
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        opt.zero_grad()
        if regression:
            y_ = model(X)
        else:
            y_, ans_ = model(X)
        loss_value = loss(y_, y)
        loss_value.backward()
        opt.step()
        
        
def get_accuracy(model, opt, loss, data_iterator):
    model.train(False)
    n = 0
    correct_pred = 0
    losses = []
    for i in data_iterator:
        x, y = i
        x = x[:, np.newaxis]
        
        X, y = torch.from_numpy(x).float(), torch.from_numpy(y)

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        y_, ans_ = model(X)
        
        loss_value = loss(y_, y)
        losses.append(loss_value.detach().clone().cpu().float())

        _, predicted_labels = torch.max(ans_, 1)

        n += y.size(0)
        correct_pred += (predicted_labels == y).sum()
    
    losses = np.array(losses).flatten()
    return correct_pred.float() / n, np.mean(losses)
    
    
def get_accuracy_regression(model, opt, loss, data_iterator, delta=0.1):
    model.train(False)
    n = 0
    correct_pred = 0
    losses = []
    for i in data_iterator:
        x, y = i
        x = x[:, np.newaxis]
        
        X, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        y_ = model(X)
        
        loss_value = loss(y_, y)
        losses.append(loss_value.detach().clone().cpu().float())

        correct_pred += (abs(y_ - y) <= delta).sum()
        n += y.size(0)
    
    losses = np.array(losses).flatten()
    return correct_pred.float() / n, np.mean(losses)
    
    
def get_error_distr(model, opt, loss, data_iterator):
    model.train(False)
    true_labels = []
    predicted_labels = []
    for i in data_iterator:
        x, y = i
        x = x[:, np.newaxis]
        true_labels.append(y)
        
        X, y = torch.from_numpy(x).float(), torch.from_numpy(y)

        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()

        y_, ans_ = model(X)
        
        _, pr_labels = torch.max(ans_, 1)

        predicted_labels.append(pr_labels.cpu().numpy())

    return np.ravel(true_labels), np.ravel(predicted_labels)
    
    
def train_net(model, opt, loss, epochs, data_iterator, test_data_iterator, path=None, regression=False):
    accuracies = []
    losses = []
    for epoch in range(epochs):
        
        train_epoch(model, opt, loss, data_iterator, regression)
        if regression:
            acc, los = get_accuracy_regression(model, opt, loss, test_data_iterator)
        else:
            acc, los = get_accuracy(model, opt, loss, test_data_iterator)
        accuracies.append(acc)
        losses.append(los)
        print(f'epoch: [{epoch+1}/{epochs}], accuracy: {acc}, loss: {los}')
        
        if not path:
            path = "HiC_" + model.name + "_e" + str(epoch) + "_" + str(datetime.date.today()) + ".tar"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'accuracies': accuracies,
            'losses': losses,
            'epochs': epochs},
        path + ".tar")

    return accuracies, losses
    
    
def plot_accuracy(acc, losses, epochs, path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10,3))
    ax.plot(np.arange(epochs)+1, np.array(acc), color='green', label='Accuracy')
    ax.legend()

    plt.xticks(np.arange(epochs)+1)
    if not path:
        plt.show()
    else:
        plt.savefig(path + "_acc.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=(10,3))
    ax.plot(np.arange(epochs)+1, np.array(losses), color='red', label='MSELoss')
    ax.legend()

    plt.xticks(np.arange(epochs)+1)
    if not path:
        plt.show()
    else:
        plt.savefig(path + "_loss.png", box_inches="tight")
