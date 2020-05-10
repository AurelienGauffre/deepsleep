from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from dataset import SoundDataset, ToTensor
from display import Bar, Color


class Metamodel:
    def __init__(
        self,
        dataset_name,
        data_nature,
        process_data,
        sampling_rate,
        transform,
        loss_fc,
        optimizer,
        learning_rate,
        nb_epochs,
        batch_size,
        test_size,
        network_class,
        scheduler,
    ):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.sampling_rate = sampling_rate
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.loss_fc = loss_fc
        self.dataset_name = dataset_name
        self.process_data = process_data
        self.learning_rate = learning_rate
        self.data_nature = data_nature
        self.test_size = test_size
        self.log_interval = 10
        self.confusion_matrix = None

        self.transform = self.init_transform(transform)
        (
            self.dataset,
            self.dataset_train,
            self.dataset_test,
        ) = self.init_dataset()
        self.train_loader, self.test_loader = self.init_data_loader()
        self.network = self.init_network(network_class)
        self.optimizer = self.init_optimizer(optimizer)
        self.scheduler = self.init_scheduler()

    def init_network(self, network_class) -> nn.Module:
        return network_class(self.dataset.nb_labels)  # network is instantiated

    def init_transform(self, transform):
        if transform is None:
            transform = []
        return transforms.Compose([ToTensor(self.data_nature)] + transform)

    def init_dataset(self) -> Tuple[torch.utils.data.Dataset, ...]:

        dataset = SoundDataset(
            dataset_name=self.dataset_name,
            transform=self.transform,
            data_nature=self.data_nature,
            sampling_rate=self.sampling_rate,
            process_data=self.process_data,
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
        self.network.to(self.device)
        y_true, y_pred = [], []
        for epoch_idx in range(self.nb_epochs):
            self.train_step(epoch_idx)
            y_true_tmp, y_pred_tmp = self.test_step(epoch_idx)
            if epoch_idx == self.nb_epochs - 1:
                y_true = y_true_tmp
                y_pred = y_pred_tmp
        print(y_true, y_pred)
        self.confusion_matrix = confusion_matrix(y_true, y_pred)

    def train_step(self, epoch_idx):
        running_loss = 0.0
        running_accuracy = 0.0
        for batch_idx, input in enumerate(Bar(self.train_loader)):
            self.optimizer.zero_grad()
            x, label = input['x'], input['label']  # TODO refactor input
            x = x.to(self.device)
            label = label.to(self.device)
            x.requires_grad_()
            output = self.network(x)
            loss = self.loss_fc(output, label)
            loss.backward()
            self.optimizer.step()

            _, pred = torch.max(output.data, 1)
            running_loss += loss.item()
            running_accuracy += torch.sum(pred == label.data)
        epoch_loss = running_loss / len(self.dataset_train)
        epoch_accuracy = running_accuracy / len(self.dataset_train)
        print(
            f'Epoch {epoch_idx}: Train acc: {epoch_accuracy * 100:.2f}% '
            f'loss: {epoch_loss:.2f}',
            end=' ',
        )

    def test_step(self, epoch_idx):
        running_loss = 0.0
        running_accuracy = 0.0
        y_pred = []
        y_true = []
        for batch_idx, input in enumerate(self.test_loader):
            with torch.no_grad():
                x, label = input['x'], input['label']
                x = x.to(self.device)
                label = label.to(self.device)
                output = self.network(x)
                loss = self.loss_fc(output, label)
                _, pred = torch.max(output.data, 1)
                running_loss += loss.item()
                y_pred += list(pred.numpy())
                y_true += list(label.data.numpy())
                running_accuracy += torch.sum(pred == label.data)
        epoch_loss = running_loss / len(self.dataset_test)
        epoch_accuracy = running_accuracy / len(self.dataset_test)
        print(
            f' Test acc: {epoch_accuracy * 100: .2f}% '
            f'loss: {epoch_loss:.2f}'
        )
        return y_true, y_pred
