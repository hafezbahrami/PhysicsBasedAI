import csv
from os import path
import matplotlib.pylab as plt
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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


def projection_matrix(
        train_data, 
        input_size = 1,
        output_size = 1,
        batch_size = 50,
        Constraint_Coefs=None
):
    if not Constraint_Coefs:
        Constraint_Coefs = torch.tensor([
            [1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0]
        ])

    Constraint_Coefs_transpose = torch.transpose(Constraint_Coefs, 0, 1)

    #Dataset variance
    Y_var = torch.zeros(output_size)
    for X, Y in train_data:
        Y_var += (batch_size)*torch.var(Y, dim=0)
    Y_var = Y_var/len(train_data.dataset.data)

    #Dataset co-variance matrix
    Error_co_var = torch.zeros(1 + input_size + output_size, 1 + input_size + output_size)
    for i in range(0, output_size):
        Error_co_var[i+input_size, i+input_size] = Y_var[i]

    #G x (Sigma x G^T)
    denominator = torch.inverse(
        torch.matmul(
            Constraint_Coefs, 
            torch.matmul(Error_co_var, Constraint_Coefs_transpose)
        )
    )

    #Projection matrix = I - (Sigma x G^T) x (denominator x G)
    Projection = torch.eye(Error_co_var.shape[0]) - torch.matmul(
            torch.matmul(Error_co_var, Constraint_Coefs_transpose), 
            torch.matmul(denominator, Constraint_Coefs)
        )
    
    G_indep = Projection[input_size:(input_size+output_size),0:input_size]
    G_dep = Projection[input_size:(input_size+output_size),input_size:(input_size+output_size)]
    Bias = Projection[input_size:(input_size+output_size), -1]
    
    return G_indep, G_dep, Bias


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
            for X1, X2, X3, Y1, Y2, Y3 in csv_reader:
                X = torch.tensor([float(X1), float(X2), float(X3)])
                Y = torch.tensor([float(Y1), float(Y2), float(Y3)])
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


class Validation:
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