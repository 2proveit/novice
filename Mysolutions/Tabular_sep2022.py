import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class MyNet(nn.Module):
    def __init__(self, feat):
        super(MyNet,self).__init__()
        self.CovdLayer = nn.Sequential(
            nn.Conv1d(1,64,3,1), # [1,1,feat]
            nn.ReLU(),
            nn.MaxPool1d(2,1,1), # [1,1,feat]

            nn.Conv1d(64,32,3,1), #
            nn.ReLU(),
            nn.AvgPool1d(2,2,1),

            nn.Conv1d(32,16,3,1),
            nn.ReLU(),
            nn.AvgPool1d(2,2,1)
        )


class TabularDataset(Dataset,):

    def __init__(self, mode,X, y=None):
        super(TabularDataset,self).__init__()
        self.mode  = mode
        if mode == 'train' or mode == "val":
            self.X = X
            self.y = y
        elif mode == 'test':
            self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        if self.mode != 'test':
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx], self.y[idx]
