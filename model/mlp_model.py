import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
            num_layers:The number of layers of the neural network (excluding the input layer). If num_layers=1, this reduces to a linear model.
            Input_dim:Dimensions of the input features
            hidden_dim:Dimensions of hidden units in all layers
            Output_dim:Number of classes to use for prediction
        '''
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.latent_dim = 25

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 1):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            for layer in range(num_layers):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            self.fc1 = nn.Linear(hidden_dim, self.latent_dim)
            self.fc2 = nn.Linear(self.latent_dim, output_dim)
    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            x = x.type(torch.float32)
            h = x
            for layer in range(self.num_layers):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            out = self.fc1(h)
            out = F.relu(out)
            out = self.fc2(out)
            out = torch.flatten(torch.sigmoid(out))
            return out

