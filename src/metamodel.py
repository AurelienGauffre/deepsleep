from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from dataset import SoundDataset, ToTensor
from display import Bar

class Metamodel:
    def __init__(
            self,
            dataset_name,
            data_nature,
            transform,
            loss_fc,
            optimizer,
            learning_rate,
            nb_epochs,
            batch_size,
            test_size,
            network=None,
            scheduler=None,
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.loss_fc = loss_fc
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.data_nature = data_nature
        self.test_size = test_size
        self.log_interval = 10

        self.transform = self.init_transform(transform)
        (
            self.dataset,
            self.dataset_train,
            self.dataset_test,
        ) = self.init_dataset()
        self.train_loader, self.test_loader = self.init_data_loader()
        self.network = self.init_network()
        self.optimizer = self.init_optimizer(optimizer)
        self.scheduler = self.init_scheduler()

    def init_network(self) -> nn.Module:
        return self.network(self.dataset.nb_labels)  # network is instantiated

    def init_transform(self, transform):
        return transforms.Compose([ToTensor(self.data_nature)] + transform)

    def init_dataset(self) -> Tuple[torch.utils.data.Dataset, ...]:

        dataset = SoundDataset(
            dataset_name=self.dataset_name,
            transform=self.transform,
            data_nature=self.data_nature,
        )
        len_test = int(len(dataset) * self.test_size)
        dataset_train, dataset_test = random_split(
            dataset, [len(dataset) - len_test, len_test]
        )
        return dataset, dataset_train, dataset_test

    def init_data_loader(self) -> Tuple[torch.utils.data.DataLoader, ...]:

        train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

        test_loader = DataLoader(
            self.dataset_test,
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
        for epoch_idx in range(self.nb_epochs):
            self.train_step(epoch_idx)
            self.test_step(epoch_idx)

    def train_step(self,epoch_idx):
        running_loss = 0
        running_acc = 0
        for batch_idx, input in enumerate(Bar(self.train_loader)):
            self.optimizer.zero_grad()
            x, label = input['sound'], input['label']
            x = x.to(self.device)
            label = label.to(self.device)
            x.requires_grad_()
            output = self.network(x)
            loss = self.loss_fc(output, label)
            loss.backward()
            self.optimizer.step()

    def test_step(self, epoch_idx):
        for batch_idx, input in enumerate(Bar(self.train_loader)):
            with torch.no_grad():
                x, label = input['sound'], input['label']
                x = x.to(self.device)
                label = label.to(self.device)
                output = self.network(x)
                loss = self.loss_fc(output, label)
