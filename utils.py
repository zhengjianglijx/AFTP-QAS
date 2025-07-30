import pickle
import torch
import torch.nn.functional as F
import numpy as np

def load_pkl(input_file):
    f = open(input_file, 'rb')
    output_file = pickle.load(f)
    f.close()
    return output_file


def save_pkl(data, loc):
    f = open(loc, 'wb')
    pickle.dump(data, file=f)
    f.close()
    return 0

def save_checkpoint_model(model, optimizer, epoch, loss, save_model):
    """Saves a checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'encoder_state': model.encoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_model)
    return 0


def save_checkpoint_model_D(model, optimizer, epoch, loss, save_model):
    """Saves a checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state': model.state_dict(),
        'encoder_1_state': model.encoder_1.state_dict(),
        'encoder_2_state': model.encoder_2.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_model)
    return 0

def normalize_adj(A):
    D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
    D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
    DA = stacked_spmm(D_in, A)  # swap D_in and D_out
    DAD = stacked_spmm(DA, D_out)
    return DAD

def preprocessing(A, H, method, lbd=None):
    # FixMe: Attention multiplying D or lbd are not friendly with the crossentropy loss in GAE
    assert A.dim()==3

    if method == 0:
        return A, H

    if method==1:
        # Adding global node with padding
        A = F.pad(A, (0,1), 'constant', 1.0)
        A = F.pad(A, (0,0,0,1), 'constant', 0.0)
        H = F.pad(H, (0,1,0,1), 'constant', 0.0 )
        H[:, -1, -1] = 1.0

    if method==1:
        # using A^T instead of A
        # and also adding a global node
        A = A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A) # swap D_in and D_out
        DAD = stacked_spmm(DA, D_out)
        return DAD, H

    elif method == 2:
        assert lbd!=None

        A = lbd * A + (1-lbd)*A.transpose(-1, -2)
        D_in = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=1)))
        D_out = torch.diag_embed(1.0 / torch.sqrt(A.sum(dim=2)))
        DA = stacked_spmm(D_in, A)
        DAD = stacked_spmm(DA, D_out)
        def prep_reverse(DAD, H):
            AD = stacked_spmm(1.0/D_in, DAD)
            A =  stacked_spmm(AD, 1.0/D_out)
            return A.triu(1), H
        return DAD, H, prep_reverse

    elif method == 3:

        assert lbd != None
        A = lbd * A + (1 - lbd) * A.transpose(-1, -2)
        def prep_reverse(A, H):
            return 1.0/lbd*A.triu(1), H
        return A, H, prep_reverse

    elif method == 4:
        A = A + A.triu(1).transpose(-1, -2)
        def prep_reverse(A, H):
            return A.triu(1), H
        return A, H, prep_reverse

    elif method == 5:
        A = A.triu(0).transpose(-1, -2)

        def prep_reverse(A, H):
            A = A.transpose(-1, -2)
            return A, H
        return A, H, prep_reverse


def get_accuracy(inputs, targets):
    N, I, _ = inputs[0].shape
    ops_recon, adj_recon = inputs[0], inputs[1]
    ops, adj = targets[0], targets[1]
    # post processing, assume non-symmetric
    adj_recon, adj = adj_recon.triu(1), adj.triu(1)
    threshold = 0.5  # hard threshold
    ops_recon_thre = ops_recon > threshold
    correct_ops = ops_recon_thre[ops.type(torch.bool)].float().sum().item() / ops.sum()
    fal_pos_ops = ops_recon_thre[~ops.type(torch.bool)].float().sum().item()/(~ops.type(torch.bool)).float().sum()
    mean_correct_adj = adj_recon[adj.type(torch.bool)].sum().item() / adj.sum()
    mean_false_positive_adj = adj_recon[(~adj.type(torch.bool)).triu(1)].sum().item() / (N*I*(I-1)/2.0-adj.sum())
    adj_recon_thre = adj_recon > threshold
    correct_adj = adj_recon_thre.eq(adj.type(torch.bool)).float().triu(1).sum().item()/ (N*I*(I-1)/2.0)

    ops_correct = ops_recon.argmax(dim=-1).eq(ops.argmax(dim=-1)).float()
    adj_correct = adj_recon_thre.eq(adj.type(torch.bool)).float()
    return correct_ops, fal_pos_ops, mean_correct_adj, mean_false_positive_adj, correct_adj


def get_train_acc(inputs, targets):
    acc_train = get_accuracy(inputs, targets)
    return 'training batch: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(*acc_train)


def get_train_NN_accuracy_str(inputs, targets, decoderNN, inds):
    acc_train = get_accuracy(inputs, targets)
    acc_val = get_NN_acc(decoderNN, targets, inds)

    return 'acc_ops:{0:.4f}({4:.4f}), mean_corr_adj:{1:.4f}({5:.4f}), mean_fal_pos_adj:{2:.4f}({6:.4f}), acc_adj:{3:.4f}({7:.4f}), top-{8} index acc {9:.4f}'.format(
        *acc_train, *acc_val)


def get_NN_acc(decoderNN, targets, inds):
    ops, adj = targets[0], targets[1]
    op_recon, adj_recon, op_recon_tk, adj_recon_tk, _, ind_tk_list = decoderNN.find_NN(ops, adj, inds)
    correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj, acc = get_accuracy((op_recon, adj_recon), targets)
    pred_k = torch.tensor(ind_tk_list,dtype=torch.int)
    correct = pred_k.eq(torch.tensor(inds, dtype=torch.int).view(-1,1).expand_as(pred_k))
    topk_acc = correct.sum(dtype=torch.float) / len(inds)
    return correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj, pred_k.shape[1], topk_acc.item()


def get_val_acc(model, cfg, X_adj, X_ops, indices):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj.cuda(), ops.cuda()
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon, _ = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave


def get_val_acc_vae1(model, cfg, X_adj, X_ops, indices):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, fal_pos_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj, ops
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, adj_recon, mu, logvar = model.forward(ops, adj)
        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, fal_pos_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon), (ops, adj))
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        fal_pos_ops_ave += fal_pos_ops * len(ind)/len(indices)
        mean_correct_adj_ave += mean_correct_adj * len(ind)/len(indices)
        mean_false_positive_adj_ave += mean_false_positive_adj * len(ind)/len(indices)
        correct_adj_ave += correct_adj * len(ind)/len(indices)

    return correct_ops_ave, fal_pos_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave


def get_val_acc_vae(model, cfg, X_adj, X_ops, indices, args):
    model.eval()
    bs = 500
    chunks = len(X_adj) // bs
    if len(X_adj) % bs > 0:
        chunks += 1
    X_adj_split = torch.split(X_adj, bs, dim=0)
    X_ops_split = torch.split(X_ops, bs, dim=0)
    indices_split = torch.split(indices, bs, dim=0)
    correct_ops_ave, acc_ave, bingos = 0, 0, 0
    for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
        adj, ops = adj.cuda(), ops.cuda()
        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])
        # forward
        ops_recon, mu, logvar = model.forward(ops, adj)
        correct_ops, bingo = get_accuracy_without_adj(ops_recon, ops)
        correct_ops_ave += correct_ops * len(ind)/len(indices)
        bingos += bingo

    return correct_ops_ave, bingos

def stacked_mm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def stacked_spmm(A, B):
    assert A.dim()==3
    assert B.dim()==3
    return torch.matmul(A, B)

def get_accuracy_without_adj(inputs, targets):
    N, I, _ = inputs.shape
    ops_recon = inputs
    ops = targets

    ops_recon1 = ops_recon[:, :, 0:9]
    ops_recon2 = ops_recon[:, :, 9:]
    ops1 = ops[:, :, 0:9]
    ops2 = ops[:, :, 9:]

    correct_ops1 = ops_recon1.argmax(dim=-1).eq(ops1.argmax(dim=-1)).float().mean().item()
    threshold = 0.5  # hard threshold
    ops_recon2_thre = ops_recon2 > threshold
    correct_ops2 = ops_recon2_thre[ops2.type(torch.bool)].float().sum().item() / ops2.sum()
    bingo = 0
    for i in range(0, len(ops)):
        ops_recon2_thre[i] = ops_recon2[i] > threshold
        condition1 = ops_recon1[i].argmax(dim=-1).eq(ops1[i].argmax(dim=-1)).float().mean().item()
        condition2 = ops_recon2_thre[i][ops2[i].type(torch.bool)].float().sum().item() / ops2[i].sum()
        if condition1 + condition2 == 2:
            bingo += 1

    return (correct_ops1+correct_ops2)/2, bingo

