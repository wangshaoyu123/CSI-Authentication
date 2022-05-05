# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:30:41 2021

@author: wangshaoyu
"""
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 20, kernel_size=(5,5),stride=(2,2))
        self.conv2 = nn.Conv2d(20, 40, kernel_size=(5,5),stride=(2,2))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(240, 80)
        self.pred = nn.Linear(80, num_classes)
        self.confidence = nn.Linear(80, 1)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 240)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        pred = self.pred(x)
        confidence = self.confidence(x)
        return  pred, confidence

