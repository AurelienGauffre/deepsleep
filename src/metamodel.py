from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from dataset import SoundDataset


class Metamodel:
    def __init__(
        self,
        dataset_name=None,
        data_nature=None,
        loss_fc=None,
        optimizer=None,
        learning_rate=None,
        nb_epochs=None,
        batch_size=None,
        scheduler=None,
        network=None,
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.log_interval = 20
        self.dataset_name = dataset_name
        self.network = network
        self.learning_rate = learning_rate
        self.data_nature = data_nature

        self.model = self.init_model()
        self.dataset = self.init_dataset()
        self.train_loader, self.test_loader = self.init_data_loader()
        self.optimizer = self.init_optimizer(optimizer)
        self.scheduler = self.init_scheduler()

    def init_model(self) -> nn.Module:
        return self.network

    def init_dataset(self) -> torch.utils.data.Dataset:
        return SoundDataset(
            dataset_name=self.dataset_name,
            transform=None,
            data_nature=self.data_nature,
        )

    def init_data_loader(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        # TODO Design a proper test data loader
        test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
        return train_loader, test_loader

    def init_optimizer(self, optimizer):
        return optimizer(self.network.parameters(), lr=self.learning_rate)

    def init_scheduler(self):
        pass

    def train(self):
        for epoch in range(self.nb_epochs):
            for batch_idx, input in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x, label = input['sound'], input['label']
                print(x)
                x = x.to(self.device)
                x.requires_grad_()
                output = self.model(x)
                # output = output.permute(
                #     1, 0, 2
                # )  # original output dimensions are batchSizex1x10
                loss = self.loss_fc(output[0], label)
                # the loss functions expects a 1xbatchSizex10 input
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.log_interval == 0:  # print training stats
                    print(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch,
                            batch_idx * len(x),
                            len(self.train_loader.dataset),
                            100.0 * batch_idx / len(self.train_loader),
                            loss,
                        )
                    )
