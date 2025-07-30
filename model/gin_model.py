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
        self.linear_or_not = False
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)

class Encoder(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(3, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(3, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, adj, ops):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        out = self.fc1(x)
        Z = torch.sum(out, dim=-2)
        return Z


class GIN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, latent_dim, mlp_num_layers, dropout):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mlp_num_layers = mlp_num_layers
        self.dropout = dropout
        self.encoder = Encoder(self.num_layers, self.input_dim, self.hidden_dim, self.latent_dim)
        self.fc = MLP(self.mlp_num_layers, self.latent_dim, self.hidden_dim, 1)

    def forward(self, adj, ops):
        batch_size, node_num, opt_num = ops.shape
        Z = self.encoder(adj, ops)
        Z_dp = F.dropout(Z, p=self.dropout, training=self.training)
        out = self.fc(Z_dp)
        out = torch.flatten(torch.sigmoid(out))
        return out, Z


class GIN_D(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, latent_dim, mlp_num_layers, dropout):
        super(GIN_D, self).__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.mlp_num_layers = mlp_num_layers
        self.dropout = dropout
        self.encoder_1 = Encoder(self.num_layers, self.input_dim, self.hidden_dim, self.latent_dim)
        self.encoder_2 = Encoder(self.num_layers, self.input_dim, self.hidden_dim, self.latent_dim)
        self.fc = MLP(self.mlp_num_layers, self.latent_dim * 2, self.hidden_dim, 1)

    def forward(self, adj_1, ops_1, adj_2=None, ops_2=None):
        batch_size, node_num, opt_num = ops_1.shape
        if adj_2 != None:
            Z1 = self.encoder_1(adj_1, ops_1)
            Z2 = self.encoder_2(adj_2, ops_2)
            Z_concat = torch.cat((Z1, Z2), dim=1)
            Z_concat = F.dropout(Z_concat, p=self.dropout, training=self.training)
            out = self.fc(Z_concat)
            out = torch.flatten(torch.sigmoid(out))
            return out, Z1
        else:
            Z1 = self.encoder_1(adj_1, ops_1)
            return Z1





