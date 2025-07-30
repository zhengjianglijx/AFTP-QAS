import torch
import torch.nn as nn
import numpy as np
from torch import optim
from utils import save_checkpoint_model, preprocessing, get_val_acc_vae, get_val_acc_vae1
from model.vae_model import VAEReconstructed_Loss1, Model_S
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import utils


class Pretraining:
    def __init__(self, args):
        self.a = 1
        self.device = 'cpu'

    def _build_dataset(self, adj, ops, ind_list):
        X_adj = []
        X_ops = []
        for ind in ind_list:
            X_adj.append(torch.Tensor(adj[ind]))
            X_ops.append(torch.Tensor(ops[ind]))
        X_adj = torch.stack(X_adj)
        X_ops = torch.stack(X_ops)
        return X_adj, X_ops, torch.Tensor(ind_list)

    def pretraining_model1(self, adj, ops, cfg, args):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        train_ind_list = [i for i in range(len(adj))]
        X_adj_train, X_ops_train, indices_train = self._build_dataset(adj, ops, train_ind_list)
        model = Model_S(input_dim=args.input_dim_pretraining, hidden_dim=args.hidden_dim_pretraining, latent_dim=args.latent_dim,
                   num_hops=args.num_layers_pretraining, num_mlp_layers=3, dropout=args.dropout_pretraining, args=args, **cfg['GAE'])

        optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
        # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        epochs = args.epochs_vae
        bs = args.train_bs_for_pretraining_vae
        loss_total = []
        train_data = TensorDataset(X_adj_train, X_ops_train)
        train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, drop_last=True)
        # best_bingo = 0
        # patient = args.pretraining_patient
        for epoch in range(0, epochs):
            model.train()
            chunks = len(train_ind_list) // bs

            loss_epoch = []
            for i, (adj, ops) in enumerate(train_loader):
                # with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                adj, ops = adj, ops
                # preprocessing
                adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
                # forward
                ops_recon, adj_recon, mu, logvar = model(ops, adj.to(torch.long))
                #     print("===")
                adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
                adj, ops = prep_reverse(adj, ops)
                loss = VAEReconstructed_Loss1(**cfg['loss'])((ops_recon, adj_recon), (ops, adj), mu, logvar, args)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                loss_epoch.append(loss.item())
            loss_total.append(sum(loss_epoch) / len(loss_epoch))

        save_model_7 = f'pretrained/{args.task}_{args.train_N_pretraining}/{args.search_space}/model_{7}_{args.loss_method}.pt'
        save_checkpoint_model(model, optimizer, epochs, loss_total[-1], save_model_7)
        return model


    def pretraining(self, adj, ops, args):
        cfg = {'GAE':
                {'activation_adj': torch.sigmoid, 'activation_ops': torch.softmax, 'adj_hidden_dim': 128, 'ops_hidden_dim': 128},
            'loss':
                {'loss_ops': nn.CrossEntropyLoss(), 'loss_adj': nn.BCELoss()},
            'prep':
                {'method': 4, 'lbd': 1.0}
            }
        # model = self.pretraining_model(arc_for_training_vae, cfg, args)
        model = self.pretraining_model1(adj, ops, cfg, args)
        return model


    def load_vae_model1(self, model_loc, args):
        cfg = {'GAE': # 4
                {'activation_adj': torch.sigmoid, 'activation_ops': torch.softmax, 'adj_hidden_dim': 128, 'ops_hidden_dim': 128},
            'loss':
                {'loss_ops': nn.CrossEntropyLoss(), 'loss_adj': nn.BCELoss()},
            'prep':
                {'method': 4, 'lbd': 1.0}
            }
        model = Model_S(input_dim=(len(args.gate_type) + args.num_qubits + 2), hidden_dim=args.hidden_dim,
                        latent_dim=args.dim,
                        num_hops=args.hops, num_mlp_layers=3, dropout=args.dropout, args=args, **cfg['GAE'])
        model.load_state_dict(torch.load(model_loc)['model_state'])
        return model

