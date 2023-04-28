import torch

class Model(torch.nn.Module):
    def __init__(
            self, 
            x_vec_size=1, 
            y_vec_size=1, 
            layers=[20, 20],
            indep_g=None, 
            dep_g=None, 
            con_bias=None
    ):
        self.x_vec_size = x_vec_size
        self.y_vec_size = y_vec_size
        self.layers = layers
        self.indep_g = indep_g
        self.dep_g = dep_g
        self.con_bias = con_bias        
        self.indep_g_transpose = torch.transpose(indep_g, 0, 1)
        super().__init__()

        L = []
        c = x_vec_size
        for l in layers:
            L.append(torch.nn.Linear(c, l))
            L.append(torch.nn.Tanh())
            c = l
        L.append(torch.nn.Linear(c, y_vec_size))
        L.append(torch.nn.Tanh())
        self.network = torch.nn.Sequential(*L)
        self.dr()
    
    def dr(self):
        self.constraint_layer = torch.nn.Linear(self.y_vec_size, self.y_vec_size)

        for name, param in self.named_parameters():
            if 'constraint_layer.weight' in name:
                param.requires_grad = False
                param.set_(self.dep_g)
        
        for name, param in self.named_parameters():
            if 'constraint_layer.bias' in name:
                param.requires_grad = False
                param.set_(self.con_bias)

    def forward(self, x):
        "make sure the x is in dimension of x_vec_size"
        y_unconstrained = self.network(x)
        y_constrained = self.constraint_layer(y_unconstrained) + torch.matmul(x, self.indep_g_transpose)
        return y_constrained