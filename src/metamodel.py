import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


class Metamodel:
    def __init__(
        self,
        dataset_name=None,
        loss_fc=None,
        optimizer=None,
        scheduler=None,
        nb_epochs=None,
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.nb_epochs = nb_epochs
        self.log_interval = 20
        self.dataset_name = None

        self.model = self.init_model()
        self.dataset = self.init_dataset()
        self.train_loader = self.init_train_loader()
        self.test_loader = self.init_test_loader()

    def init_model(self) -> nn.Module:
        pass

    def init_dataset(self) -> torch.utils.data.Dataset:
        # TO COMPLETE HERE
        pass

    def init_train_loader(self) -> torch.utils.data.DataLoader:
        pass

    def init_test_loader(self) -> torch.utils.data.DataLoader:
        pass

    def train(self):
        log_interval = 10
        for epoch in range(1, self.nb_epochs + 1):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                data = (
                    data.requires_grad_()
                )  # set requires_grad to True for training
                output = self.model(data)
                output = output.permute(
                    1, 0, 2
                )  # original output dimensions are batchSizex1x10
                loss = self.loss_fc(
                    output[0], target
                )  # the loss functions expects a 1xbatchSizex10 input
                loss.backward()
                self.optimizer.step()
                if batch_idx % log_interval == 0:  # print training stats
                    print(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch,
                            batch_idx * len(data),
                            len(self.train_loader.dataset),
                            100.0 * batch_idx / len(self.train_loader),
                            loss,
                        )
                    )
