import torch


def __init__(self, layers=[16, 32, 64, 128], n_input_channels=3, n_output_channels=6, kernel_size=5):
    super().__init__()

    L = []
    c = n_input_channels
    for l in layers:
        L.append(torch.nn.Conv2d(c, l, kernel_size, stride=2, padding=kernel_size // 2))
        L.append(torch.nn.ReLU())
        c = l
    self.network = torch.nn.Sequential(*L)
    self.classifier = torch.nn.Linear(c, n_output_channels)


class LinearModel(torch.nn.Module):
    def __init__(self, x_in_vec_size=2, y_vec_size=1, layers=[5]):
        super().__init__()
        L = []
        c = x_in_vec_size
        for l in layers:
            L.append(torch.nn.Linear(c, l))
            L.append(torch.nn.ReLU())
            c = l
        self.intermediate = torch.nn.Sequential(*L)
        self.network = torch.nn.Linear(c, y_vec_size)

    def forward(self, x):
        "make sure the x is in dimension of (B, x_vec_size"
        return self.network(self.intermediate(x))