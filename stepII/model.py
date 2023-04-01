import torch

class Model(torch.nn.Module):
    def __init__(self, x_vec_size=5, y_vec_size=4, layers=[20, 20]):
        super().__init__()

        L = []
        c = x_vec_size
        for l in layers:
            L.append(torch.nn.Linear(c, l))
            L.append(torch.nn.Tanh())
            c = l
        L.append(torch.nn.Linear(c, y_vec_size))
        self.network = torch.nn.Sequential(*L)

    def forward(self, x):
        "make sure the x is in dimension of x_vec_size"
        return self.network(x)