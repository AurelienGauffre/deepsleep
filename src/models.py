import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from metamodel import Metamodel


class NetA0(nn.Module):
    def __init__(self):
        super(NetA0, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, 80, 4)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(
            30
        )  # input should be 512x30 so this outputs a 512x1
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = self.avgPool(x)
        x = x.permute(0, 2, 1)  # change the 512x1 to 1x512
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


class ModelA(Metamodel):
    def __init__(
        self,
        dataset_name=None,
        data_nature='1D',
        loss_fc=nn.NLLLoss(),
        optimizer=torch.optim.Adamax,
        learning_rate=10e-3,
        nb_epochs=10,
        batch_size=4,
        scheduler=None,
        network=NetA0(),
    ):
        self.network = network
        super().__init__(
            dataset_name,
            data_nature,
            loss_fc,
            optimizer,
            learning_rate,
            nb_epochs,
            batch_size,
            scheduler,
            network,
        )
