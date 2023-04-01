import csv
from os import path
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MixerDataset(Dataset):
    def __init__(self, dataset_path="./"):
        self.data = []
        self.dataset_path = dataset_path
        self.debug_mod = True

        to_tensor = transforms.ToTensor()

        if self.debug_mod:
            abs_path = path.abspath(__file__)
            dir_path = path.dirname(abs_path)
            file_path = path.join(dir_path, self.dataset_path)
            dataset_location = path.join(file_path, "input.csv")
        else:
            dataset_location = path.join(self.dataset_path, "input.csv")

        with open(dataset_location, newline="") as f:
            csv_reader = csv.reader(f)
            for X1, X2 in csv_reader:
                X = torch.tensor([float(X1), float(X2)])
                self.data.append(X)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(dataset_path, batch_size=128, shuffle=True):
    dataset = MixerDataset(dataset_path=dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



def accuracy(y_pred, y_label):
    acc = torch.sqrt(torch.mean((y_pred - y_label) ** 2))
    return acc.detach().numpy().item()