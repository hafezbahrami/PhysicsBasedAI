import csv
from os import path
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
torch.set_default_dtype(torch.float64)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MixerDataset(Dataset):
    def __init__(self, dataset_path="./", file_name="input_data.csv", transformX=None, transformY=None
    ):
        self.data = []
        self.dataset_path = dataset_path
        self.debug_mod = True
        self.file_name = file_name

        dataset_location = self._helper(file_name)
        with open(dataset_location, newline="") as f:
            csv_reader = csv.reader(f)
            for X1, X2, X3, X4, X5, Y1, Y2, Y3, Y4, Y5, Y6, Y7, _, X_syn1, X_syn2, X_syn3, X_syn4, X_syn5 in csv_reader:
                X = torch.tensor([
                    float(X1), float(X2), float(X3), float(X4), float(X5), 
                    float(X_syn1), float(X_syn2), float(X_syn3), float(X_syn4), float(X_syn5), 
                ])
                Y = torch.tensor([
                    float(Y1), float(Y2), float(Y3), float(Y4), float(Y5), float(Y6), float(Y7),
                ])
                if transformX and transformY:
                    X = transformX(X[None, :, None, None]).squeeze()
                    Y = transformY(Y[None, :, None, None]).squeeze()
                self.data.append((X, Y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _helper(self, file_name: str):
        if self.debug_mod:
            abs_path = path.abspath(__file__)
            dir_path = path.dirname(abs_path)
            file_path = path.join(dir_path, self.dataset_path)
            dataset_location = path.join(file_path, file_name)
        else:
            dataset_location = path.join(self.dataset_path, file_name)
        return dataset_location


def load_data_train(
        dataset_path="./",
        file_name="input_data.csv",
        batch_size=100,
        Shuffle_flag=True,
        transformX=None,
        transformY=None,
        ):
    dataset = MixerDataset(dataset_path=dataset_path, file_name=file_name, transformX=transformX, transformY=transformY)
    data_loader_data = DataLoader(dataset, batch_size, shuffle=Shuffle_flag)
    return data_loader_data


def rmse(y_pred, y_label):
    rmse_val = torch.sqrt(torch.mean((y_pred - y_label) ** 2))
    return rmse_val.detach().numpy().item()


def preprocessing(dataset_path="./", file_name="input_data.csv"):
    """preprocessing data to calculate the mean and std"""
    dataset_location = path.join(dataset_path, file_name)
    data = pd.read_csv(dataset_location)
    avg_data = torch.tensor(data.mean())
    std_data = torch.tensor(data.std())
    min_data = torch.tensor(data.min())
    max_data = torch.tensor(data.max())
    return min_data, max_data, avg_data, std_data


class ValidationPlot:
    @staticmethod
    def plot_unity(y_pred, y_actual):
        plt.figure(figsize=(8,7), dpi=120)
        plt.plot(np.linspace(min(y_actual), 1.2*max(y_actual), 101), np.linspace(min(y_actual), 1.2*max(y_actual), 101), '--',
                 color=(0, 0, 0, 1), linewidth=3.5)
        plt.plot(y_pred, y_actual, 'go', markerfacecolor='r', markersize=12.5)
        plt.xlabel("Prediction", fontname="Times New Roman", fontsize=25)
        plt.ylabel("Actual", fontname="Times New Roman", fontsize=25)
        plt.xticks(fontname="Times New Roman", fontsize=20)
        plt.yticks(fontname="Times New Roman", fontsize=20)
        plt.show()

    @staticmethod
    def plot_unity_w_constraint(y_con, y_pred, y_actual):
        plt.figure(figsize=(8,7), dpi=120)
        plt.plot(np.linspace(min(y_actual), 1.2*max(y_actual), 101), np.linspace(min(y_actual), 1.2*max(y_actual), 101), '--',
                 color=(0, 0, 0, 1), linewidth=3.5)
        plt.plot(y_pred, y_actual, 'go', markerfacecolor='r', markersize=12.5)
        plt.vlines(y_con, min(y_actual), 1.2*max(y_actual), linestyles='--', color=(0, 0, 0, 1), linewidth=3.5)
        plt.xlabel("Prediction", fontname="Times New Roman", fontsize=25)
        plt.ylabel("Actual", fontname="Times New Roman", fontsize=25)
        plt.xticks(fontname="Times New Roman", fontsize=20)
        plt.yticks(fontname="Times New Roman", fontsize=20)
        plt.show()

    @staticmethod
    def plot_non_equality(greater, lesser):
        plt.figure(figsize=(8,7), dpi=120)
        plt.plot(np.zeros(len(greater)), '--',
                 color=(0, 0, 0, 1), linewidth=3.5)
        plt.plot(greater - lesser, 'go', markerfacecolor='r', markersize=12.5)
        # plt.xlabel("Prediction", fontname="Times New Roman", fontsize=25)
        # plt.ylabel("Actual", fontname="Times New Roman", fontsize=25)
        plt.xticks([], fontname="Times New Roman", fontsize=20)
        plt.yticks(fontname="Times New Roman", fontsize=20)
        plt.show()

    @staticmethod
    def plot_regular(x_con, x, y_act, y_pred):
        plt.figure(figsize=(8,7), dpi=120)
        plt.plot(x, y_act , 's', markerfacecolor='b', markersize=12.5)
        plt.plot(x, y_pred, 'o', markerfacecolor='g', markersize=10.0)
        plt.hlines(x_con, min(x), max(x), linestyles='--', color=(0, 0, 0, 1), linewidth=3.5)
        plt.xlabel("X Variable", fontname="Times New Roman", fontsize=25)
        plt.ylabel("Y Variable", fontname="Times New Roman", fontsize=25)
        plt.xticks(fontname="Times New Roman", fontsize=20)
        plt.yticks(fontname="Times New Roman", fontsize=20)
        plt.legend({'Prediction', 'Actual'}, prop={'family':"Times New Roman", 'size':25})
        plt.show()

