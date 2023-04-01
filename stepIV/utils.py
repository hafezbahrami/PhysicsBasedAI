import csv
from os import path
import matplotlib.pylab as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class MixerDataset(Dataset):
    def __init__(self, dataset_path="./", file_name="input_data.csv",
    ):
        self.data = []
        self.dataset_path = dataset_path
        self.debug_mod = True
        self.file_name = file_name

        dataset_location = self._helper(file_name)
        with open(dataset_location, newline="") as f:
            csv_reader = csv.reader(f)
            for X1, Y1, _, X_syn_1, in csv_reader:
                X = torch.tensor([float(X1), float(X_syn_1)])
                Y = torch.tensor([float(Y1)])
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
        Shuffle_flag=True
        ):
    dataset = MixerDataset(dataset_path=dataset_path, file_name=file_name)
    data_loader_data = DataLoader(dataset, batch_size, shuffle=Shuffle_flag)
    return data_loader_data


def rmse(y_pred, y_label):
    rmse_val = torch.sqrt(torch.mean((y_pred - y_label) ** 2))
    return rmse_val.detach().numpy().item()


class Validation:
    @staticmethod
    def plot_unity(y_pred, y_actual):
        plt.figure(figsize=(8,7), dpi=120)
        plt.plot(np.linspace(0, 1.2*max(y_actual), 101), np.linspace(0, 1.2*max(y_actual), 101), '--',
                 color=(0, 0, 0, 1), linewidth=3.5)
        plt.plot(y_pred, y_actual, 'go', markerfacecolor='r', markersize=12.5)
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
        plt.plot(x, y_pred, 'o', markerfacecolor='g', markersize=10)
        plt.hlines(x_con, min(x), max(x), linestyles='--', color=(0, 0, 0, 1), linewidth=3.5)
        plt.xlabel("X Variable", fontname="Times New Roman", fontsize=25)
        plt.ylabel("Y Variable", fontname="Times New Roman", fontsize=25)
        plt.xticks(fontname="Times New Roman", fontsize=20)
        plt.yticks(fontname="Times New Roman", fontsize=20)
        plt.legend({'Prediction', 'Actual'}, prop={'family':"Times New Roman", 'size':25})
        plt.show()

