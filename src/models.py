import cooler

import torch
from torch import nn, optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import datetime



"""
Filters and iterates over chromosomes
Returns upper right subimage, i.e. the subimage which changes the most and stores the main information of the distance
dist 0 - (n_classes-1) : class (i / (n_classes-1))  \in [0, 1]
"""
class DataIteratorFiltered_TopRight_Regression:
  """
  Batch_size = n_classes * bs
  """
  def __init__(self, data, sz, bs, n_classes, chrs, delta=0):
    self.data = data
    self.chrs = chrs
    self.bs = bs
    self.sz = sz
    self.i = 0
    self.n_classes = n_classes
    self.delta = delta
    
    self.cur_chr = 0
    self.cur_data = self.data.matrix().fetch(self.chrs[self.cur_chr])
    self.cur_n = self.cur_data.shape[0]


  def __iter__(self):
    self.i = 0
    self.cur_chr = 0
    self.cur_data = self.data.matrix().fetch(self.chrs[self.cur_chr])
    self.cur_n = self.cur_data.shape[0]
    self.total = 0
    self.good = 0
    self.good_cl = [0] * self.n_classes
    return self

  def __next__(self):
    while True:
      if self.i + self.bs - 1 + 2 * self.sz + (self.delta+1)*self.n_classes > self.cur_n:
        if self.cur_chr == len(self.chrs)-1:
          print("total =", self.total, ", good =", self.good, ", good_classes =", self.good_cl)
          raise StopIteration
        else:
          #print(self.chrs[self.cur_chr])
          self.cur_chr += 1
          self.cur_data = self.data.matrix().fetch(self.chrs[self.cur_chr])
          self.cur_n = self.cur_data.shape[0]
          self.i = 0

      
      self.current = np.empty(shape=(self.n_classes * self.bs, self.sz, self.sz), dtype=np.float64)
      self.ans = np.zeros(shape=(self.n_classes * self.bs), dtype=np.float64)
      pos = 0
      for a in range(self.i, self.i + self.bs):
        for t in range(0, (self.delta+1)*self.n_classes, self.delta+1):
          self.ans[pos] = t//(self.delta+1) / (self.n_classes - 1)
          self.current[pos] = np.nan_to_num(self.cur_data[a : a + self.sz, a + self.sz + t : a + 2*self.sz + t])
          # check whether >=1000/1024 cells are informative
          if np.count_nonzero(self.current[pos]) >= 1000:
              pos += 1
              self.good_cl[t//(self.delta+1)] += 1
      self.i += self.bs
      
      self.ans = self.ans[:pos]
      self.current = self.current[:pos]
      self.good += pos
      self.total += self.n_classes * self.bs
      if pos > 0:
        break
    # shuffle
    rng_state = np.random.get_state()
    np.random.shuffle(self.current)
    np.random.set_state(rng_state)
    np.random.shuffle(self.ans)
    return self.current, self.ans


"""
Original LeNet for 32x32 images and regression output
"""
class LeNet_regression(nn.Module):

  def __init__(self):
    super(LeNet_regression, self).__init__()

    self.name = "LeNet_regression"
    
    self.feature_extractor = nn.Sequential(
        # 1x32x32 -> 6x28x28           
        nn.Conv2d(1, 6, (5, 5), stride=1, padding=0),
        nn.ReLU(),
        # 6x28x28 -> 6x14x14           
        nn.MaxPool2d(kernel_size=2),
        # 6x14x14 -> 16x10x10           
        nn.Conv2d(6, 16, (5, 5), stride=1, padding=0),
        nn.ReLU(),
        # 16x10x10 -> 16x5x5         
        nn.MaxPool2d(kernel_size=2),
        # 16x5x5 -> 120x1x1         
        nn.Conv2d(16, 120, (5, 5), stride=1, padding=0),
        nn.ReLU()
    )

    # regression
    self.regressor = nn.Sequential(
        nn.Linear(in_features=120, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=1),
        nn.Sigmoid()
    )


  def forward(self, x):
    x = self.feature_extractor(x)
    x = torch.flatten(x, 1)
    ans = self.regressor(x)
    ans = torch.flatten(ans)
    return ans