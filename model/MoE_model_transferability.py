import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gin_model import GIN, GIN_D
from model.vae_model import Encoder_S

# Gating networks
class GatingNetwork(nn.Module):
    def __init__(self, arg):
        super(GatingNetwork, self).__init__()
        self.encoder = Encoder_S(arg.input_dim_pretraining, arg.hidden_dim_pretraining, arg.latent_dim, arg.num_layers_pretraining, 3)
        self.encoder.load_state_dict(
            torch.load(f'pretrained/{arg.task}_{arg.train_N_pretraining}/{arg.search_space_tran}/model_7_{arg.loss_method}.pt')['encoder_state'])
        # Define the gating network
        self.gate = nn.Sequential(
            nn.Linear(arg.latent_dim, arg.hidden_dim),
            nn.BatchNorm1d(arg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=arg.dropout_pretraining),
            nn.Linear(arg.hidden_dim, arg.num_experts)
        )

    def forward(self, adj, ops):
        Z, logvar = self.encoder(ops, adj)
        Z = torch.sum(Z, dim=-2)
        gating_scores = self.gate(Z)
        gating_weights = F.softmax(gating_scores, dim=1)
        return gating_weights


# Expert networks
class Expert(nn.Module):
    def __init__(self, arg):
        super(Expert, self).__init__()
        # Initialize 5 GIN experts and 1 GIN_D expert
        self.experts = nn.ModuleList([
            GIN(arg.num_layers_pretraining, arg.input_dim_pretraining, arg.hidden_dim_pretraining, arg.latent_dim,
                arg.mlp_num_layers, arg.dropout_pretraining) for _ in range(5)])
        self.expert_6 = GIN_D(arg.num_layers_pretraining, arg.input_dim_pretraining, arg.hidden_dim_pretraining, arg.latent_dim, arg.mlp_num_layers, arg.dropout_pretraining)

        self.fc6 = nn.Sequential(
            nn.Linear(arg.latent_dim, arg.hidden_dim),
            nn.BatchNorm1d(arg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=arg.dropout_pretraining),
            nn.Linear(arg.hidden_dim, arg.output_dim),
            nn.Sigmoid()
        )
        for i, expert in enumerate(self.experts, start=1):
            expert.load_state_dict(
                torch.load(f'pretrained/{arg.task}_{arg.train_N_pretraining}/{arg.search_space_tran}/model_{i}_{arg.loss_method}.pt')['model_state'])
        self.expert_6.load_state_dict(
            torch.load(f'pretrained/{arg.task}_{arg.train_N_pretraining}/{arg.search_space_tran}/model_6_{arg.loss_method}.pt')['model_state'])

    def forward(self, adj, ops):
        exp_scores = torch.stack([expert(adj, ops)[0] for expert in self.experts], dim=1)
        exp_score_6 = self.fc6(self.expert_6(adj, ops)).squeeze(-1).unsqueeze(1)
        return torch.cat([exp_scores, exp_score_6], dim=1)

# Mixture-of-Experts (MoE)
class MoE_tran(nn.Module):
    def __init__(self, arg):
        super(MoE_tran, self).__init__()
        self.gating = GatingNetwork(arg)
        self.experts = Expert(arg)

        # Freeze GIN parameters in GatingNetwork
        for param in self.gating.encoder.parameters():
            param.requires_grad = False

        # Freeze the encoder parameters in the expert network
        for expert in self.experts.experts:
            for param in expert.encoder.parameters():
                param.requires_grad = False
        for param in self.experts.expert_6.encoder_1.parameters():
            param.requires_grad = False
        for param in self.experts.expert_6.encoder_2.parameters():
            param.requires_grad = False

    def forward(self, adj, ops):
        gating_weights = self.gating(adj, ops)
        expert_scores = self.experts(adj, ops)
        output = torch.sum(gating_weights * expert_scores, dim=1)
        return output, gating_weights


