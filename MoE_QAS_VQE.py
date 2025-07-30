"""
QAS, Start!
"""
import os
import utils
import numpy as np
from decimal import Decimal, getcontext
import argparse
from task_configs import configs
import time
from architecture_generator import ArchitectureGenerator
from torch.utils.data import TensorDataset, DataLoader
from optimize_circuits.vqe_task_paralell import VqeTrainerNew
import tensorcircuit as tc
from model.MoE_model import MoE
from model.MoE_model_transferability import MoE_tran
from model.gin_model import GIN, GIN_D
from model.vae_pretraining import Pretraining
from utils import save_checkpoint_model, save_checkpoint_model_D
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

tc.set_backend("tensorflow")
tc.set_dtype("complex128")

def get_one_hot(data_batch):
    mapping = {
        (1, 1): [1, 0, 0, 0],
        (1, -1): [0, 1, 0, 0],
        (-1, 1): [0, 0, 1, 0],
        (-1, -1): [0, 0, 0, 1]
    }
    data_one_hot = [mapping.get(tuple(data)) for data in data_batch]
    return np.array(data_one_hot).astype(np.float64)

def data_divide_MoE(list_cir, pad_adj, pad_ops, arg):
    np.random.seed(arg.seed)
    data = list(zip(list_cir, pad_adj, pad_ops))
    np.random.shuffle(data)
    data_train = data[:arg.train_N][:arg.num_initial_training_set]
    data_test = data[arg.train_N:]
    return data_train, data_test

def data_divide_VQE(cir_list, arg):
    np.random.seed(arg.seed)
    data = cir_list
    np.random.shuffle(data)
    data_train = data[:arg.train_N][:arg.num_initial_training_set]
    data_test = data[arg.train_N:]
    return data_train, data_test

def data_divide(cir_list, arg):
    np.random.seed(arg.seed)
    data = cir_list
    np.random.shuffle(data)
    data_train = data[:arg.train_N][:arg.num_initial_training_set]
    data_test = data[arg.train_N:]
    return data_train, data_test

def pad_adj_ops(adj_list, ops_list, max_nodes):
    padded_adj_list = []
    padded_ops_list = []
    for adj, ops in zip(adj_list, ops_list):
        num_nodes = adj.shape[0]

        padded_adj = np.zeros((max_nodes, max_nodes))
        padded_adj[:num_nodes, :num_nodes] = adj

        padded_ops = np.zeros((max_nodes, ops.shape[1]))
        padded_ops[:num_nodes, :] = ops
        padded_adj_list.append(padded_adj)
        padded_ops_list.append(padded_ops)

    return np.stack(padded_adj_list), np.stack(padded_ops_list)

def levenshtein_distance(s1, s2):
    '''
    :param s1: list of Gate objects, a sequence of gates
    :param s2: list of Gate objects, another sequence of gates
    :return: int, Levenshtein distance between two sequences of gates
    '''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize the boundary conditions
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    # fill matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):

            gate_s1 = s1[i - 1]
            gate_s2 = s2[j - 1]
            if (gate_s1.name == gate_s2.name and
                gate_s1.qubits == gate_s2.qubits and
                gate_s1.para_gate == gate_s2.para_gate and
                gate_s1.act_on == gate_s2.act_on):
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,    # delete
                           dp[i][j - 1] + 1,    # Insert
                           dp[i - 1][j - 1] + cost)  # replace
    return dp[m][n]


class BPRLoss(torch.nn.Module):

    def __init__(self, exp_weighted=False):
        super(BPRLoss, self).__init__()
        self.exp_weighted = exp_weighted

    def forward(self, input, target):
        N = input.size(0)
        total_loss = 0
        for i in range(N):
            indices = (target > target[i])
            x = torch.log(1 + torch.exp(-(input[indices] - input[i])))
            if self.exp_weighted:
                x = (torch.exp(target[i]) - 1) * (torch.exp(target[indices]) - 1) * x
            else:
                x = x
            total_loss += torch.sum(x)
        if self.exp_weighted:
            return 2 / (N * (np.exp(1) - 1))**2 * total_loss
        else:
            return 2 / N**2 * total_loss

def sort_normalization(proxy_list):
    m, M = proxy_list.shape
    proxy_scores = np.argsort(np.argsort(proxy_list, axis=0), axis=0).astype(float) + 1
    proxy_rank = proxy_scores / m
    return proxy_rank

def depth_count(cir, qubit):
    res = [0] * qubit
    for gate in cir:
        if gate.qubits > 1:
            depth_q = []
            for q in gate.act_on:
                depth_q.append(res[q])
            max_depth = max(depth_q)
            max_depth += 1
            for q in gate.act_on:
                res[q] = max_depth
        else:
            res[gate.act_on[0]] += 1
    depth = np.max(res)
    return depth

def linear_ranking_aggregation(list_cir, proxy_values_test, arg):
    _, data_test = data_divide(list_cir, arg)
    list_cir_test = data_test
    list_cir_test = np.array(list_cir_test)
    proxy_values_test = np.array(proxy_values_test)
    comp_score = []
    M = len(proxy_values_test)
    m = len(proxy_values_test[0])
    proxy_rank = []
    for i in range(M):
        proxy_scores = [Decimal(str(score)) for score in proxy_values_test[i]]
        ranked = np.argsort(np.argsort(proxy_scores))
        proxy_rank.append((ranked + 1) / m)

    for j in range(m):
        temp_score = 0
        for i in range(M):
            temp_score += proxy_rank[i][j]
        comp_score.append(temp_score)
    rank = np.argsort(-np.array(comp_score))
    return list_cir_test, rank, comp_score

def non_linear_ranking_aggregation(list_cir, proxy_values_test, arg):
    _, data_test = data_divide(list_cir, arg)
    list_cir_test = data_test
    list_cir_test = np.array(list_cir_test)
    proxy_values_test = np.array(proxy_values_test)
    comp_score = []
    M = len(proxy_values_test)
    m = len(proxy_values_test[0])
    proxy_rank = []
    for i in range(M):
        proxy_scores = [Decimal(str(score)) for score in proxy_values_test[i]]
        ranked = np.argsort(np.argsort(proxy_scores))
        proxy_rank.append(ranked + 1)
    for j in range(m):
        temp_score = 0
        for i in range(M):
            temp_score += np.log(proxy_rank[i][j]/m)
        comp_score.append(temp_score)
    rank = np.argsort(-np.array(comp_score))
    return list_cir_test, rank, comp_score

def single_proxy(list_cir, proxy_values_test, arg):
    _, data_test = data_divide(list_cir, arg)
    list_cir_test = data_test
    list_cir_test = np.array(list_cir_test)
    proxy_values_test = np.array(proxy_values_test)
    M = len(proxy_values_test)
    proxy_rank = []
    for i in range(M):
        proxy_scores = [Decimal(str(score)) for score in proxy_values_test[i]]
        rank = np.argsort(-1 * np.array(proxy_scores))
        proxy_rank.append(rank)
    return list_cir_test, proxy_rank

def compute_edit_distance(list_cir_1, list_cir_2):
    edit_distance = [levenshtein_distance(list_cir_1[i], list_cir_2[i]) for i in range(0, len(list_cir_1))]
    return np.argsort(np.argsort(edit_distance)).astype(float) / len(edit_distance)

def pretrain_model(model, optimizer, loss_func, train_loader, proxy_column, epochs, save_model, is_gin_d=False):
    total_loss = []
    # Set the learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch in train_loader:
            if is_gin_d:
                b_adj_1, b_ops_1, b_adj_2, b_ops_2, b_distance = batch
                target = b_distance
                forward, _ = model(b_adj_1, b_ops_1, b_adj_2, b_ops_2)
            else:
                b_adj, b_ops, b_proxy = batch
                target = b_proxy[:, proxy_column]
                forward, _ = model(b_adj, b_ops)
            optimizer.zero_grad()
            loss = loss_func(forward, target)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        avg_loss = np.mean(train_loss)
        total_loss.append(avg_loss)
        scheduler.step(avg_loss)

    if is_gin_d:
        save_checkpoint_model_D(model, optimizer, epochs, total_loss[-1], save_model)
    else:
        save_checkpoint_model(model, optimizer, epochs, total_loss[-1], save_model)

    return total_loss

def Pretraining_expert(matrix_cir, list_cir, proxy_values, arg, cfg, save_result_model_pretained):
    proxy_SN = sort_normalization(np.array(proxy_values).T)
    adj = np.array([matrix_cir[i][0] for i in range(len(matrix_cir))])
    ops = np.array([matrix_cir[i][1] for i in range(len(matrix_cir))])
    max_nodes = max([len(ops[i]) for i in range(len(ops))])
    pad_adj, pad_ops = pad_adj_ops(adj, ops, max_nodes)

    list_cir_train = list_cir[:arg.train_N_pretraining]
    pad_adj_train = pad_adj[:arg.train_N_pretraining]
    pad_ops_train = pad_ops[:arg.train_N_pretraining]
    proxy_SN_train = proxy_SN[:arg.train_N_pretraining]

    adj_train, ops_train, proxy_train = torch.Tensor(pad_adj_train), torch.Tensor(pad_ops_train), torch.Tensor(proxy_SN_train)
    train_loader = DataLoader(TensorDataset(adj_train, ops_train, proxy_train), batch_size=arg.train_bs_for_pretraining, shuffle=True, drop_last=False)
    loss_list = []

    for i in range(1, 6):
        print(f'Pretraining Model {i} start...')
        model = GIN(num_layers=arg.num_layers_pretraining, input_dim=arg.input_dim_pretraining, hidden_dim=arg.hidden_dim_pretraining, latent_dim=arg.latent_dim, mlp_num_layers=arg.mlp_num_layers, dropout=arg.dropout_pretraining)
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.lr_for_pretraining)
        loss_func = BPRLoss()

        save_model = save_result_model_pretained + f'model_{i}_{arg.loss_method}.pt'
        total_loss = pretrain_model(model, optimizer, loss_func, train_loader, i - 1, arg.pretraining_epochs, save_model)
        loss_list.append(total_loss)

    print('Pretraining Model 6 (GIN_D) start...')
    adj_1_train, adj_2_train = adj_train[:len(adj_train) // 2], adj_train[len(adj_train) // 2:]
    ops_1_train, ops_2_train = ops_train[:len(ops_train) // 2], ops_train[len(ops_train) // 2:]
    list_cir_1_train, list_cir_2_train = list_cir_train[:len(list_cir_train) // 2], list_cir_train[len(list_cir_train) // 2:]

    edit_distance_train = torch.Tensor(compute_edit_distance(list_cir_1_train, list_cir_2_train))

    train_loader_d = DataLoader(TensorDataset(adj_1_train, ops_1_train, adj_2_train, ops_2_train, edit_distance_train), batch_size=arg.train_bs_for_pretraining, shuffle=True, drop_last=False)

    model_6 = GIN_D(num_layers=arg.num_layers_pretraining, input_dim=arg.input_dim_pretraining, hidden_dim=arg.hidden_dim_pretraining, latent_dim=arg.latent_dim, mlp_num_layers=arg.mlp_num_layers, dropout=arg.dropout_pretraining)
    optimizer_6 = torch.optim.Adam(model_6.parameters(), lr=arg.lr_for_pretraining)
    loss_func_6 = BPRLoss()

    save_model_6 = save_result_model_pretained + f'model_{6}_{arg.loss_method}.pt'
    total_loss_6 = pretrain_model(model_6, optimizer_6, loss_func_6, train_loader_d, 0, arg.pretraining_epochs, save_model_6, is_gin_d=True)
    loss_list.append(total_loss_6)

    print('------------------------')
    return 0

def Pretraining_vae(matrix_cir, arg):
    adj = np.array([matrix_cir[i][0] for i in range(len(matrix_cir))])
    ops = np.array([matrix_cir[i][1] for i in range(len(matrix_cir))])
    max_nodes = max([len(ops[i]) for i in range(len(ops))])
    pad_adj, pad_ops = pad_adj_ops(adj, ops, max_nodes)
    pretrainer = Pretraining(arg)
    vae_model = pretrainer.pretraining(pad_adj, pad_ops, arg)
    vae_model.eval()
    return 0

def MoE_search_test(matrix_cir, list_cir, labels_train, arg, cfg):
    adj = np.array([matrix_cir[i][0] for i in range(len(matrix_cir))])
    ops = np.array([matrix_cir[i][1] for i in range(len(matrix_cir))])
    max_nodes = max([len(ops[i]) for i in range(len(ops))])
    pad_adj, pad_ops = pad_adj_ops(adj, ops, max_nodes)
    data_train, data_test = data_divide_MoE(list_cir, pad_adj, pad_ops, arg)
    list_cir_train, adj_train, ops_train = zip(*data_train)
    list_cir_test, adj_test, ops_test = zip(*data_test)
    llist_cir_train, adj_train, ops_train, labels_train = np.array(list_cir_train), np.array(adj_train), np.array(ops_train), np.array(labels_train)
    list_cir_test, adj_test, ops_test = np.array(list_cir_test), np.array(adj_test), np.array(ops_test)
    labels_train = labels_train[:arg.num_initial_training_set]
    labels_train = (labels_train - np.min(labels_train)) / (np.max(labels_train) - np.min(labels_train))

    adj_train = torch.Tensor(adj_train)
    ops_train = torch.Tensor(ops_train)
    labels_train = torch.Tensor(labels_train)

    adj_test = torch.Tensor(adj_test)
    ops_test = torch.Tensor(ops_test)

    train_data = TensorDataset(adj_train, ops_train, labels_train)
    test_data = TensorDataset(adj_test, ops_test)
    train_loader = DataLoader(train_data, batch_size=arg.train_bs_for_predictor, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=arg.test_bs_for_predictor, shuffle=False, drop_last=False)
    MoE_predictor = MoE(arg)
    optimizer = torch.optim.Adam(MoE_predictor.parameters(), lr=arg.lr_for_predictor)
    loss_func = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    total_loss = []

    for epoch in range(arg.pre_epochs):
        MoE_predictor.train()
        train_loss = []
        for step, (b_adj, b_ops, b_y) in enumerate(train_loader):
            optimizer.zero_grad()
            b_adj, b_ops = b_adj, b_ops
            forward, _ = MoE_predictor(b_adj, b_ops)
            loss = loss_func(forward, b_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        total_loss.append(train_loss)
        scheduler.step(train_loss)

    pred = []
    MoE_predictor.eval()
    print('iter:%d:predicting energy...')
    for step, (b_adj, b_ops) in enumerate(test_loader):
        b_adj, b_ops = b_adj, b_ops
        with torch.no_grad():
            p, _ = MoE_predictor(b_adj, b_ops)
            p = p.detach().cpu().tolist()
            pred.extend(p)
    print('predicting complete')
    print('ranking...')
    rank = np.argsort(np.array(pred))
    return list_cir_test, rank, pred


def MoE_search_test_removal(matrix_cir, list_cir, labels_train, arg, cfg):
    adj = np.array([matrix_cir[i][0] for i in range(len(matrix_cir))])
    ops = np.array([matrix_cir[i][1] for i in range(len(matrix_cir))])
    max_nodes = max([len(ops[i]) for i in range(len(ops))])
    pad_adj, pad_ops = pad_adj_ops(adj, ops, max_nodes)
    data_train, data_test = data_divide_MoE(list_cir, pad_adj, pad_ops, arg)
    list_cir_train, adj_train, ops_train = zip(*data_train)
    list_cir_test, adj_test, ops_test = zip(*data_test)
    llist_cir_train, adj_train, ops_train, labels_train = np.array(list_cir_train), np.array(adj_train), np.array(ops_train), np.array(labels_train)
    list_cir_test, adj_test, ops_test = np.array(list_cir_test), np.array(adj_test), np.array(ops_test)

    labels_train = labels_train[:arg.num_initial_training_set]
    labels_train = (labels_train - np.min(labels_train)) / (np.max(labels_train) - np.min(labels_train))

    adj_train = torch.Tensor(adj_train)
    ops_train = torch.Tensor(ops_train)
    labels_train = torch.Tensor(labels_train)

    adj_test = torch.Tensor(adj_test)
    ops_test = torch.Tensor(ops_test)

    train_data = TensorDataset(adj_train, ops_train, labels_train)
    test_data = TensorDataset(adj_test, ops_test)
    train_loader = DataLoader(train_data, batch_size=arg.train_bs_for_predictor, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=arg.test_bs_for_predictor, shuffle=False, drop_last=False)
    MoE_predictor = MoE(arg)
    optimizer = torch.optim.Adam(MoE_predictor.parameters(), lr=arg.lr_for_predictor)
    loss_func = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    total_loss = []

    for epoch in range(arg.pre_epochs):
        MoE_predictor.train()
        train_loss = []
        for step, (b_adj, b_ops, b_y) in enumerate(train_loader):
            optimizer.zero_grad()
            b_adj, b_ops = b_adj, b_ops
            forward, _ = MoE_predictor.forward_ablation(b_adj, b_ops, arg.expert_to_remove)
            loss = loss_func(forward, b_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        total_loss.append(train_loss)
        scheduler.step(train_loss)

    pred = []
    MoE_predictor.eval()
    print('iter:%d:predicting energy...')
    for step, (b_adj, b_ops) in enumerate(test_loader):
        b_adj, b_ops = b_adj, b_ops
        with torch.no_grad():
            p, _ = MoE_predictor.forward_ablation(b_adj, b_ops, arg.expert_to_remove)
            p = p.detach().cpu().tolist()
            pred.extend(p)
    print('predicting complete')

    print('ranking...')
    rank = np.argsort(np.array(pred))

    return list_cir_test, rank, pred


def MoE_search_test_tran(matrix_cir, list_cir, labels_train, arg, cfg):
    adj = np.array([matrix_cir[i][0] for i in range(len(matrix_cir))])
    ops = np.array([matrix_cir[i][1] for i in range(len(matrix_cir))])
    max_nodes = max([len(ops[i]) for i in range(len(ops))])
    pad_adj, pad_ops = pad_adj_ops(adj, ops, max_nodes)
    data_train, data_test = data_divide_MoE(list_cir, pad_adj, pad_ops, arg)
    list_cir_train, adj_train, ops_train = zip(*data_train)
    list_cir_test, adj_test, ops_test = zip(*data_test)
    llist_cir_train, adj_train, ops_train, labels_train = np.array(list_cir_train), np.array(adj_train), np.array(ops_train), np.array(labels_train)
    list_cir_test, adj_test, ops_test = np.array(list_cir_test), np.array(adj_test), np.array(ops_test)

    labels_train = labels_train[:arg.num_initial_training_set]
    labels_train = (labels_train - np.min(labels_train)) / (np.max(labels_train) - np.min(labels_train))
    adj_train = torch.Tensor(adj_train)
    ops_train = torch.Tensor(ops_train)
    labels_train = torch.Tensor(labels_train)

    adj_test = torch.Tensor(adj_test)
    ops_test = torch.Tensor(ops_test)

    train_data = TensorDataset(adj_train, ops_train, labels_train)
    test_data = TensorDataset(adj_test, ops_test)
    train_loader = DataLoader(train_data, batch_size=arg.train_bs_for_predictor, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=arg.test_bs_for_predictor, shuffle=False, drop_last=False)
    MoE_predictor = MoE_tran(arg)
    optimizer = torch.optim.Adam(MoE_predictor.parameters(), lr=arg.lr_for_predictor)
    loss_func = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    total_loss = []

    for epoch in range(arg.pre_epochs):
        MoE_predictor.train()
        train_loss = []
        for step, (b_adj, b_ops, b_y) in enumerate(train_loader):
            optimizer.zero_grad()
            b_adj, b_ops = b_adj, b_ops
            forward, _ = MoE_predictor(b_adj, b_ops)
            loss = loss_func(forward, b_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        total_loss.append(train_loss)
        scheduler.step(train_loss)

    pred = []
    MoE_predictor.eval()
    print('iter:%d:predicting energy...')
    for step, (b_adj, b_ops) in enumerate(test_loader):
        b_adj, b_ops = b_adj, b_ops
        with torch.no_grad():
            p, _ = MoE_predictor(b_adj, b_ops)
            p = p.detach().cpu().tolist()
            pred.extend(p)
    print('predicting complete')

    print('ranking...')
    rank = np.argsort(np.array(pred))

    return list_cir_test, rank, pred


def main(arg, cfg):
    start_time = time.time()
    np.random.seed(arg.seed)
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    argsDict = arg.__dict__
    print(f'searching information:')
    for key, value in argsDict.items():
        print(f'{key}:{value}')
    print('---------------------------')
    print(f'task information:')
    for key, value in cfg.items():
        print(f'{key}:{value}')
    time.sleep(0.1)

    ag = ArchitectureGenerator(gate_pool=cfg['gate_pool'],
                               two_gate_num_range=arg.two_gate_num_range,
                               num_layers=cfg['layer'],
                               num_qubits=cfg['qubit'],
                               not_first=cfg['not_first'],
                               start_with_u=cfg['start_with_u'],
                               num_embed_gates=16)

    save_datasets = f'./datasets/{arg.task}/'
    save_result_data = f'./result/{arg.task_result}/{arg.method}/{arg.search_space}/{arg.train_N_pretraining}_{arg.num_initial_training_set}/data/{arg.seed}/'
    save_result_data_S = f'./result/{arg.task_result}/{arg.method}/{arg.search_space}/data/{arg.seed}/'
    save_result_model_pretained = f'pretrained/{arg.task}_{arg.train_N_pretraining}/{arg.search_space}/'
    save_result_data_removal = f'./result/{arg.task_result}/{arg.method}/{arg.search_space}/{arg.train_N_pretraining}_{arg.num_initial_training_set}/{arg.expert_to_remove}/data/{arg.seed}/'
    save_result_data_tran = f'./result/{arg.task_result}/{arg.method}/{arg.search_space}/{arg.train_N_pretraining}_{arg.num_initial_training_set}/{arg.search_space_tran}/data/{arg.seed}/'


    if args.pretraining:
        print('Start of pre-training')
        if not os.path.exists(save_result_model_pretained):
            os.makedirs(save_result_model_pretained)
        s1 = time.time()
        circuit_path = f'{save_datasets}/{arg.search_space}_seed{arg.seed}_pretrained/'
        list_cir_pretrained = utils.load_pkl(circuit_path + f'list_cir_{arg.search_space}_seed{arg.seed}_pretrained/list_arc_{arg.search_space}_seed{arg.seed}_pretrained.pkl')
        list_cir_pretrained_vae = utils.load_pkl(circuit_path + f'list_cir_{arg.search_space}_seed{arg.seed}_pretrained_vae/list_arc_{arg.search_space}_seed{arg.seed}_pretrained_vae.pkl')

        if os.path.exists(circuit_path + f'list_cir_{arg.search_space}_seed{arg.seed}_pretrained/matrix_pretrained.pkl'):
            matrix_cir_pretrained = utils.load_pkl(circuit_path + f'list_cir_{arg.search_space}_seed{arg.seed}_pretrained/matrix_pretrained.pkl')
            matrix_cir_pretrained_vae = utils.load_pkl(circuit_path + f'list_cir_{arg.search_space}_seed{arg.seed}_pretrained_vae/matrix_pretrained_vae.pkl')
        else:
            print('list to adj...')
            matrix_cir_pretrained = ag.list_to_adj(list_cir_pretrained)
            matrix_cir_pretrained_vae = ag.list_to_adj(list_cir_pretrained_vae)
            for i in range(0, len(matrix_cir_pretrained)):
                matrix_cir_pretrained[i][0] = np.logical_or(matrix_cir_pretrained[i][0], matrix_cir_pretrained[i][0].T).astype(int)
            utils.save_pkl(matrix_cir_pretrained, circuit_path + f'list_cir_{arg.search_space}_seed{arg.seed}_pretrained/matrix_pretrained.pkl')
            utils.save_pkl(matrix_cir_pretrained_vae, circuit_path + f'list_cir_{arg.search_space}_seed{arg.seed}_pretrained_vae/matrix_pretrained_vae.pkl')

        depth_avg_pretrained = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}_pretrained/Depth/depth_number.pkl')
        width_pretrained = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}_pretrained/Path/path_number.pkl')
        expressibility_pretrained = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}_pretrained/EX/expressibility.pkl')
        trainability_pretrained = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}_pretrained/Tr_task_w/trainability.pkl')
        snip_pretrained = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}_pretrained/Snip_task_w/snip.pkl')

        proxy_list_pretrained = [expressibility_pretrained, trainability_pretrained, snip_pretrained, width_pretrained, depth_avg_pretrained]
        Pretraining_expert(matrix_cir_pretrained, list_cir_pretrained, proxy_list_pretrained, arg, cfg, save_result_model_pretained)
        Pretraining_vae(matrix_cir_pretrained_vae, arg)
        s2 = time.time()
        run_time = s2 - s1
        print(run_time)
        print('End of pre-training')
        exit(0)
    circuit_path = f'{save_datasets}/{arg.search_space}_seed{arg.seed}/list_cir_{arg.search_space}_seed{arg.seed}/'

    '''Loading task data'''
    if ag.start_with_u:
        ag.start_gate = cfg['start_gate']

    list_cir = utils.load_pkl(circuit_path + f'list_arc_{arg.search_space}_seed{arg.seed}.pkl')
    depth_avg = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}/Depth/depth_number.pkl')
    width = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}/Path/path_number.pkl')
    expressibility = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}/EX/expressibility.pkl')
    trainability = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}/Tr_task_w/trainability.pkl')
    snip = utils.load_pkl(f'{save_datasets}/{arg.search_space}_seed{arg.seed}/Snip_task_w/snip.pkl')

    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    if arg.task == 'VQE':
        trainer = VqeTrainerNew(task_configs=cfg, n_cir_parallel=arg.parallel, n_runs=arg.run,
                                max_iteration=arg.iteration, noise_param=noise_param)
    else:
        trainer = None
        print('invalid task')
        exit(0)

    if arg.task == 'VQE':
        data_train, data_test = data_divide_VQE(list_cir, arg)
        list_cir_train = data_train
        energy_result, param_result, energy_epoch = trainer.process(list_cir_train)
        labels_train = energy_result

    """"Method"""
    if arg.method == 'Single_proxy':
        print('list to adj...')
        if not os.path.exists(save_result_data_S):
            os.makedirs(save_result_data_S)
        proxy_list_1 = [expressibility, trainability, snip, width, depth_avg]
        list_cir_test, rank_list = single_proxy(list_cir, proxy_list_1, arg)
        index_eval = []
        list_cir_eval = []
        for i, rank_row in enumerate(rank_list):
            index_eval_temp = rank_row[:arg.query_num]
            list_cir_eval_temp = [list_cir_test[idx] for idx in index_eval_temp]
            index_eval.extend(index_eval_temp)
            list_cir_eval.extend(list_cir_eval_temp)
        utils.save_pkl(index_eval, f'{save_result_data_S}/index_eval.pkl')
        utils.save_pkl(list_cir_eval, f'{save_result_data_S}/list_cir_eval.pkl')
        exit(0)

    elif arg.method == 'Liner_QAS':
        if not os.path.exists(save_result_data_S):
            os.makedirs(save_result_data_S)
        proxy_list_1 = [expressibility, trainability, snip, width, depth_avg]
        list_cir_test, rank, comp_score = linear_ranking_aggregation(list_cir, proxy_list_1, arg)
        index_eval = []
        list_cir_eval = []
        for i in range(arg.query_num):
            index_eval.append(rank[i])
            list_cir_eval.append(list_cir_test[rank[i]])
        utils.save_pkl(index_eval, f'{save_result_data_S}/index_eval.pkl')
        utils.save_pkl(list_cir_eval, f'{save_result_data_S}/list_cir_eval.pkl')
        exit(0)

    elif arg.method == 'NonLiner_QAS':
        if not os.path.exists(save_result_data_S):
            os.makedirs(save_result_data_S)
        proxy_list_1 = [expressibility, trainability, snip, width, depth_avg]
        list_cir_test, rank, comp_score = non_linear_ranking_aggregation(list_cir, proxy_list_1, arg)
        index_eval = []
        list_cir_eval = []
        for i in range(arg.query_num):
            index_eval.append(rank[i])
            list_cir_eval.append(list_cir_test[rank[i]])
        utils.save_pkl(index_eval, f'{save_result_data_S}/index_eval.pkl')
        utils.save_pkl(list_cir_eval, f'{save_result_data_S}/list_cir_eval.pkl')
        exit(0)

    elif arg.method == 'MoE_QAS':
        print('list to adj...')
        if not os.path.exists(save_result_data):
            os.makedirs(save_result_data)
        with open(save_result_data + '/setting.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in cfg.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('-------------------------------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
        f.close()
        matrix_cir = ag.list_to_adj(list_cir)
        for i in range(0, len(matrix_cir)):
            matrix_cir[i][0] = np.logical_or(matrix_cir[i][0], matrix_cir[i][0].T).astype(int)
        print(len(matrix_cir))
        list_cir_test, rank, pred = MoE_search_test(matrix_cir, list_cir, labels_train, arg, cfg)
        index_eval = []
        list_cir_eval = []
        for i in range(arg.query_num):
            index_eval.append(rank[i])
            list_cir_eval.append(list_cir_test[rank[i]])
        utils.save_pkl(index_eval, f'{save_result_data}/index_eval.pkl')
        utils.save_pkl(list_cir_eval, f'{save_result_data}/list_cir_eval.pkl')
        exit(0)

    elif arg.method == 'MoE_QAS_removal':
        print('list to adj...')
        if not os.path.exists(save_result_data_removal):
            os.makedirs(save_result_data_removal)
        with open(save_result_data_removal + '/setting.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in cfg.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('-------------------------------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
        f.close()
        matrix_cir = ag.list_to_adj(list_cir)
        for i in range(0, len(matrix_cir)):
            matrix_cir[i][0] = np.logical_or(matrix_cir[i][0], matrix_cir[i][0].T).astype(int)
        print(len(matrix_cir))
        list_cir_test, rank, pred = MoE_search_test_removal(matrix_cir, list_cir, labels_train, arg, cfg)
        index_eval = []
        list_cir_eval = []
        for i in range(arg.query_num):
            index_eval.append(rank[i])
            list_cir_eval.append(list_cir_test[rank[i]])
        utils.save_pkl(index_eval, f'{save_result_data_removal}/index_eval.pkl')
        utils.save_pkl(list_cir_eval, f'{save_result_data_removal}/list_cir_eval.pkl')
        exit(0)

    else:
        print('invalid method')
        exit(0)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--task', type=str, default='VQE', help='see task_congigs.py for available tasks')
    parser.add_argument('--task_result', type=str, default='TFIM', help='')
    parser.add_argument('--search_space', type=str, default='layerwise', help='')
    parser.add_argument('--method', type=str, default='MoE_QAS', help='search method')
    parser.add_argument('--qubit', type=int, default=7, help='')
    parser.add_argument('--pretraining', type=bool, default=False, help='')
    parser.add_argument('--max_gate_num', type=int, default=64, help='')
    parser.add_argument('--gate_type', type=list, default=['CNOT', 'Ry', 'Rz', 'Rx'])
    parser.add_argument('--two_gate_num_range', type=list, default=[6, 10])
    parser.add_argument('--num_experts', type=int, default=6, help='')
    parser.add_argument('--search_space_tran', type=str, default='gatewise', help='')
    parser.add_argument('--query_num', type=int, default=1, help='')

    """pretraining setting """
    parser.add_argument("--train_N_pretraining", type=int, default=5000, help="")
    parser.add_argument('--train_bs_for_pretraining', type=int, default=64, help='batch size for training')
    parser.add_argument('--train_bs_for_pretraining_vae', type=int, default=32, help='batch size for training')
    parser.add_argument('--test_bs_for_pretraining', type=int, default=100, help='batch size')
    parser.add_argument('--lr_for_pretraining', type=float, default=0.001, help='learning rate')
    parser.add_argument("--pretraining_epochs", type=int, default=100, help="")
    parser.add_argument('--epochs_vae', type=int, default=5)
    parser.add_argument("--num_layers_pretraining", type=int, default=5, help="")
    parser.add_argument("--input_dim_pretraining", type=int, default=13, help="")
    parser.add_argument("--hidden_dim_pretraining", type=int, default=128, help="")
    parser.add_argument('--dropout_pretraining', type=float, default=0.3, help='decoder implicit regularization (default: 0.3)')

    """training setting """
    parser.add_argument('--loss_method', type=str, default='B', help='')
    parser.add_argument("--train_N", type=int, default=1000, help="")
    parser.add_argument('--num_initial_training_set', type=int, default=100, help='')
    parser.add_argument("--mlp_num_layers", type=int, default=2, help="")
    parser.add_argument("--input_dim", type=int, default=13, help="")
    parser.add_argument("--output_dim", type=int, default=1, help="")
    parser.add_argument("--hidden_dim", type=int, default=128, help="")
    parser.add_argument("--latent_dim", type=int, default=16, help="")

    parser.add_argument("--pre_epochs", type=int, default=40, help="")
    parser.add_argument('--train_bs_for_predictor', type=int, default=16, help='batch size for training')
    parser.add_argument('--test_bs_for_predictor', type=int, default=100, help='batch size')
    parser.add_argument('--lr_for_predictor', type=float, default=0.001, help='learning rate')

    """quantum setting """
    parser.add_argument("--run", type=int, default=10, help="number of repetitions per line")
    parser.add_argument('--batch_size', default=200, type=int, help='batch size for training')
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train the generated circuits for QNN")
    parser.add_argument("--iteration", type=int, default=2000, help="number of epochs to train the generated circuits for VQE")
    parser.add_argument("--parallel", type=int, default=10, help="multiprocessing training circuit")
    parser.add_argument("--train_lr", type=float, default=0.01, help="learning rate for training")
    parser.add_argument("--min_delta", type=float, default=0.001, help="early stopping parameters")
    parser.add_argument("--patience", type=int, default=6, help="early stopping parameters")

    parser.add_argument("--noise", type=bool, default=True, help="whether to consider noise")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01, help="magnitude of the two-qubit depolarization noise")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001, help="magnitude of depolarization noise for a single qubit")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="bit-flip noise magnitude")
    args = parser.parse_args()
    cfgs = configs[args.task_result]
    main(args, cfgs)
