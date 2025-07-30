import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vae_mlp_model import MLP


class VAEReconstructed_Loss1(object):
    def __init__(self, w_ops=1.0, w_adj=1.0, loss_ops=None, loss_adj=None):
        super().__init__()
        self.w_ops = w_ops
        self.w_adj = w_adj
        self.loss_ops = loss_ops
        self.loss_adj = loss_adj

    def __call__(self, inputs, targets, mu, logvar, args):
        ops_recon, adj_recon = inputs[0], inputs[1]
        ops, adj = targets[0], targets[1]
        break_point = args.input_dim_pretraining - args.qubit
        loss_ops1 = self.loss_ops(ops_recon[:, :, 0:break_point].view(-1, break_point), ops[:, :, 0:break_point].view(-1, break_point))
        loss_ops2 = self.loss_adj(ops_recon[:, :, break_point:], ops[:, :, break_point:])
        loss_adj = self.loss_adj(adj_recon, adj)
        loss = self.w_ops * (loss_ops1+loss_ops2) + self.w_adj * loss_adj
        KLD = -0.5 / (ops.shape[0] * ops.shape[1]) * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 2))
        return loss + KLD


class Decoder_S(nn.Module):
    def __init__(self, embedding_dim, input_dim, dropout, args, activation_adj=torch.sigmoid, activation_ops=torch.sigmoid, adj_hidden_dim=None, ops_hidden_dim=None):
        super(Decoder_S, self).__init__()

        self.sigmoid = torch.sigmoid
        self.softmax = torch.softmax
        self.gate_num = args.max_gate_num + 2
        self.fcn1 = torch.nn.Linear(embedding_dim, len(args.gate_type)+2)
        self.fcn2 = torch.nn.Linear(embedding_dim, args.qubit)
        self.fcn3 = torch.nn.Linear(self.gate_num, self.gate_num)
        self.dropout = dropout

    def forward(self, embedding):
        embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        adj = torch.matmul(embedding, embedding.permute(0, 2, 1))
        adj = self.fcn3(adj)
        type = self.softmax(self.fcn1(embedding), dim=-1)
        qubit = self.sigmoid(self.fcn2(embedding))
        ops = torch.cat((type, qubit), -1)

        return ops, self.sigmoid(adj)


class Encoder_S(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers):
        super(Encoder_S, self).__init__()
        self.num_layers = num_hops
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.fc1 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, ops, adj):
        batch_size, node_num, opt_num = ops.shape
        x = ops
        for l in range(self.num_layers - 1):
            neighbor = torch.matmul(adj.float(), x)
            agg = (1 + self.eps[l]) * x.view(batch_size * node_num, -1) + neighbor.view(batch_size * node_num, -1)
            x = F.relu(self.batch_norms[l](self.mlps[l](agg)).view(batch_size, node_num, -1))
        mu = self.fc1(x)
        logvar = self.fc2(x)

        return mu, logvar


class Model_S(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers,
                 dropout, args, **kwargs):
        super(Model_S, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_hops
        self.encoder = Encoder_S(input_dim, hidden_dim, latent_dim, num_hops, num_mlp_layers)
        self.decoder = Decoder_S(self.latent_dim, self.input_dim, dropout, args, **kwargs)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, ops, adj):
        mu, logvar = self.encoder(ops, adj)
        z = self.reparameterize(mu, logvar)
        ops_recon, adj_recon = self.decoder(z)
        return ops_recon, adj_recon, mu, logvar


