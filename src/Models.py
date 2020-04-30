import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class Model0():
    def __init__(self, dataset=None,optimizer=None,scheduler=None,nb_epochs=None):
        self.model = self.init_model()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.nb_epochs = nb_epochs
        self.log_interval = 20
        self.train_loader = None

    def init_model(self) -> nn.Module:
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
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
                self.avgPool = nn.AvgPool1d(30)  # input should be 512x30 so this outputs a 512x1
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

        return Model()

    def train(self):
        for epoch in range(1, nb_epochs + 1):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                data = data.to(device)
                target = target.to(device)
                data = data.requires_grad_()  # set requires_grad to True for training
                output = self.model(data)
                output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
                loss = self.loss_fc(output[0], target)  # the loss functions expects a 1xbatchSizex10 input
                loss.backward()
                self.optimizer.step()
                if batch_idx % log_interval == 0:  # print training stats
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss))


Model0 = Model0("nameofdataset")
