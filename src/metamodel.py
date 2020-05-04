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
        self.loss_fc = loss_fc
        self.log_interval = 2
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.data_nature = data_nature

        self.dataset = self.init_dataset()
        self.network = self.init_network()
        self.train_loader, self.test_loader = self.init_data_loader()
        self.optimizer = self.init_optimizer(optimizer)
        self.scheduler = self.init_scheduler()

    def init_network(self) -> nn.Module:
        return self.network(self.dataset.nb_labels)  # we instantiate the
        # network here

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
                x = x.to(self.device)
                label = label.to(self.device)
                x.requires_grad_()
                output = self.network(x)
                # original output dimensions are batchSizex1x10
                loss = self.loss_fc(output, label)
                # the loss functions expects a 1xbatchSizex10 input
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.log_interval == 0:  # print training stats
                    print(
                        f'Train Epoch: {epoch} [{batch_idx * len(x)}'
                        f'{len(self.train_loader.dataset)} '
                        f'({100.0 * batch_idx / len(self.train_loader)}%)]'
                        f'\tLoss: {loss:.2f}'
                    )
