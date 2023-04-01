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
            for X1, X2, X3, X4, Y1, Y2, _, X_syn_1, X_syn_2, X_syn_3, X_syn_4, in csv_reader:
                X = torch.tensor([float(X1), float(X2), float(X3), float(X4),
                                  float(X_syn_1), float(X_syn_2), float(X_syn_3), float(X_syn_4)])
                Y = torch.tensor([float(Y1), float(Y2)])
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
        plt.figure()
        plt.plot(np.linspace(0, 1.2*max(y_actual), 101), np.linspace(0, 1.2*max(y_actual), 101), '--',
                 color=(0, 0, 0, 1), linewidth=3.5)
        plt.plot(y_pred, y_actual, 'go', markerfacecolor='r', markersize=12.5)
        plt.xlabel("Prediction", fontname="Times New Roman", fontsize=25)
        plt.ylabel("Actual", fontname="Times New Roman", fontsize=25)
        plt.xticks(fontname="Times New Roman", fontsize=20)
        plt.yticks(fontname="Times New Roman", fontsize=20)
        plt.show()

