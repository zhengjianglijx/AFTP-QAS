import argparse
import os
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import utils
import torch
from multiprocessing import Pool
import matplotlib.pyplot as plt
from data_process_T1 import TensorFlowDataset
from optimize_circuits.classify_task_paralell import ClassifyTrainerNew
import gc

from task_configs import configs

tc.set_dtype("complex128")
tc.set_backend("tensorflow")


def get_one_hot(data_batch):
    mapping = {
        (1, 1): [1, 0, 0, 0],
        (1, -1): [0, 1, 0, 0],
        (-1, 1): [0, 0, 1, 0],
        (-1, -1): [0, 0, 0, 1]
    }
    data_one_hot = [mapping.get(tuple(data)) for data in data_batch]
    return np.array(data_one_hot).astype(np.float64)


def batch_acc(preds, labels):
    # Convert predictions and labels from one-hot to category index
    preds_class = np.argmax(preds, axis=1)
    labels_class = np.argmax(labels, axis=1)
    # Count the number of correct predictions
    correct_predictions = np.sum(preds_class == labels_class)
    accuracy = correct_predictions / len(labels)
    return accuracy

class ClassifyTrainerNew_prune:
    def __init__(self, arg, n_qubit, data_class, noise_param=None):
        self.n_qubit = n_qubit
        self.parallel = arg.parallel
        self.num_epochs = arg.num_epochs
        self.prune_check_num = arg.prune_check_num
        self.softmax_flag = arg.softmax_flag
        self.seed = 0
        self.give_up_rest = False
        self.solution = None
        self.data_class = data_class
        self.batch_size = arg.batch_size
        self.train_lr = arg.train_lr
        self.min_delta = arg.min_delta
        self.patience = arg.patience

        if noise_param is None:
            self.noise = False
        else:
            self.noise = True
        self.two_qubit_channel_depolarizing_p = None
        self.single_qubit_channel_depolarizing_p = None
        self.bit_flip_p = None
        if self.noise:
            self.two_qubit_channel_depolarizing_p = noise_param['two_qubit_channel_depolarizing_p']
            self.single_qubit_channel_depolarizing_p = noise_param['single_qubit_channel_depolarizing_p']
            self.bit_flip_p = noise_param['bit_flip_p']
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.two_qubit_channel_depolarizing_p / 15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.single_qubit_channel_depolarizing_p / 3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

    def depth_count(self, cir, qubit):
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

    def compute_energy(self, x, y, param, structure, embed_flags, param_visit):
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        y = tf.convert_to_tensor(y, dtype=tf.float64)
        param = tf.convert_to_tensor(param, dtype=tf.float64)
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)
            c = tc.DMCircuit(self.n_qubit)
            x_index = 0
            param_index = 0
            for i, gate in enumerate(structure):
                if embed_flags[i]:
                    if gate.name == "Ry":
                        c.ry(gate.act_on[0], theta=x[x_index])
                        c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                        x_index += 1
                    elif gate.name == "Rz":
                        c.rz(gate.act_on[0], theta=x[x_index])
                        c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                        x_index += 1
                    elif gate.name == "Rx":
                        c.rx(gate.act_on[0], theta=x[x_index])
                        c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                        x_index += 1
                    else:
                        print("invalid gate!")
                        exit(0)
                else:
                    if gate.name == "CNOT":
                        c.cnot(gate.act_on[0], gate.act_on[1])
                        c.general_kraus(self.two_qubit_dep_channel, gate.act_on[0], gate.act_on[1])
                    elif gate.name == "Ry":
                        if param_visit[param_index]:
                            c.ry(gate.act_on[0], theta=param[param_index])
                            c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                        param_index += 1
                    elif gate.name == "Rz":
                        if param_visit[param_index]:
                            c.rz(gate.act_on[0], theta=param[param_index])
                            c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                        param_index += 1
                    elif gate.name == "Rx":
                        if param_visit[param_index]:
                            c.rx(gate.act_on[0], theta=param[param_index])
                            c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                        param_index += 1
                    else:
                        print("invalid gate!")
                        exit(0)
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)
        else:
            c = tc.Circuit(self.n_qubit)
            x_index = 0
            param_index = 0
            for i, gate in enumerate(structure):
                if embed_flags[i]:
                    if gate.name == "Ry":
                        c.ry(gate.act_on[0], theta=x[x_index])
                        x_index += 1
                    elif gate.name == "Rz":
                        c.rz(gate.act_on[0], theta=x[x_index])
                        x_index += 1
                    elif gate.name == "Rx":
                        c.rx(gate.act_on[0], theta=x[x_index])
                        x_index += 1
                    else:
                        print("invalid gate!")
                        exit(0)
                else:
                    if gate.name == "CNOT":
                        c.cnot(gate.act_on[0], gate.act_on[1])
                    elif gate.name == "Ry":
                        c.ry(gate.act_on[0], theta=param[param_index])
                        param_index += 1
                    elif gate.name == "Rz":
                        c.rz(gate.act_on[0], theta=param[param_index])
                        param_index += 1
                    elif gate.name == "Rx":
                        c.rx(gate.act_on[0], theta=param[param_index])
                        param_index += 1
                    else:
                        print("invalid gate!")
                        exit(0)
        ypreds = []
        for i in range(self.n_qubit):
            ypreds.append(c.expectation_ps(z=[i]))
        ypreds = tc.backend.real(ypreds)
        ypreds_sm = tc.backend.softmax(ypreds)
        loss = tf.keras.losses.categorical_crossentropy(y, ypreds_sm)
        return loss, ypreds_sm

    def generate_param_emb_index(self, arc, inputs_bounds, weights_bounds, embed_flags, param_flags):
        for i in range(len(arc)):
            if weights_bounds[i] != weights_bounds[i + 1]:
                embed_flags.append(0)
                param_flags.append(1)
            else:
                param_flags.append(0)
                if inputs_bounds[i] != inputs_bounds[i + 1]:
                    embed_flags.append(1)
                else:
                    embed_flags.append(0)
        return embed_flags, param_flags

    def get_qml_vvag(self):
        qml_vvag_train = tc.backend.vectorized_value_and_grad(self.compute_energy, argnums=(2,), vectorized_argnums=(0, 1), has_aux=True)
        qml_vvag_test = tc.backend.vmap(self.compute_energy, vectorized_argnums=(0, 1))
        qml_vvag_train = tc.backend.jit(qml_vvag_train, static_argnums=(3, 4))
        qml_vvag_test = tc.backend.jit(qml_vvag_test, static_argnums=(3, 4))
        return qml_vvag_train, qml_vvag_test

    def build_saliency(self, grad_buffer, thresh_vec, param_visit):
        # gbuf = grad_buffer.flatten()
        gbuf = grad_buffer
        prune_idx = []
        for i, g in enumerate(gbuf):
            if g < thresh_vec[i] and param_visit[i]:
                prune_idx.append(i)
        return prune_idx

    def train_circuit(self, work_que):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        tc.set_backend("tensorflow")
        embed_flags = []
        param_flags = []
        total_loss = []
        total_acc = []
        test_acc = []
        single_circuit, inputs_bounds, weights_bounds = work_que[0], work_que[1], work_que[2]
        x_train = self.data_class[0]
        y_train = self.data_class[1]
        x_test = self.data_class[2]
        y_test = self.data_class[3]

        x_train_tensor = tf.convert_to_tensor(x_train)
        y_train_tensor = tf.convert_to_tensor(y_train)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train_tensor, y_train_tensor))
        train_data_loader = train_dataset.shuffle(buffer_size=len(x_train)).batch(self.batch_size)

        embed_flags, param_flags = self.generate_param_emb_index(single_circuit, inputs_bounds, weights_bounds, embed_flags, param_flags)
        param_gate_index = []
        for i in range(len(param_flags)):
            if param_flags[i]:
                param_gate_index.append(i)
        L = self.depth_count(single_circuit, self.n_qubit)
        par = np.random.normal(loc=0, scale=1 / (4 * self.n_qubit * (L + 2)), size=weights_bounds[-1])  # 4S * (L+2)
        param = tf.Variable(initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr)))
        qml_vvag_train, qml_vvag_test = self.get_qml_vvag()
        opt = tf.keras.optimizers.Adam(self.train_lr)

        # Early stopping parameters
        patience = self.patience
        min_delta = self.min_delta
        best_loss = float('inf')
        patience_counter = 0
        grad_list = []
        prune_index = []
        prune_information = []
        train_loss = 0
        train_ypreds = []
        tau = [1 / weights_bounds[-1]] * weights_bounds[-1]
        param_visit = [True] * weights_bounds[-1]
        for step, (xs, ys) in enumerate(train_data_loader):
            (loss, ypreds), grad = qml_vvag_train(xs, ys, param, single_circuit, embed_flags, param_visit)
            grad = tf.squeeze(grad)
            opt.apply_gradients([(grad, param)])
            ypreds_numpy = [ypreds[i].numpy() for i in range(len(ypreds))]
            train_ypreds.extend(ypreds_numpy)
            train_loss += tc.backend.sum(loss).numpy()
        total_loss.append(train_loss / len(x_train))
        total_acc.append(batch_acc(train_ypreds, y_train))
        gbuffer = grad
        grad_list.append(grad)

        for epoch in range(self.num_epochs):
            if self.softmax_flag:
                tau = tau - tau * tf.nn.softmax(grad_list[-1])
            else:
                tau = tau - tau * abs(grad_list[-1])
            train_loss = 0
            train_ypreds = []
            for step, (xs, ys) in enumerate(train_data_loader):
                # print(f'step: {step}')
                (loss, ypreds), grad = qml_vvag_train(xs, ys, param, single_circuit, embed_flags, param_visit)
                grad = tf.squeeze(grad)
                opt.apply_gradients([(grad, param)])
                ypreds_numpy = [ypreds[i].numpy() for i in range(len(ypreds))]
                train_ypreds.extend(ypreds_numpy)
                train_loss += tc.backend.sum(loss).numpy()
            total_loss.append(train_loss / len(x_train))
            total_acc.append(batch_acc(train_ypreds, y_train))
            grad_list.append(grad)
            gbuffer = gbuffer + abs(grad_list[-2] - grad_list[-1])

            if epoch != 0 and (epoch + 1) % self.prune_check_num == 0:
                idx = self.build_saliency(gbuffer.numpy(), tau.numpy(), param_visit)
                # print(f"Dropping {idx} for threshold {tau}")
                for i in idx:
                    param_visit[i] = False
                prune_index.extend(idx)
                gbuffer = grad
                # print(f"Step: {epoch}, Prune: {prune_index}")
                prune_information.append({'Step': epoch, 'Prune': prune_index, 'param_gate_index': param_gate_index})
            # Early stopping logic
            if abs(total_loss[epoch] - best_loss) < min_delta:
                patience_counter += 1
            else:
                best_loss = total_loss[epoch]
                patience_counter = 0

            if patience_counter >= patience:
                break

        loss_test, ypreds_test = qml_vvag_test(x_test, y_test, param, single_circuit, embed_flags, param_visit)
        ypreds_test_numpy = [ypreds_test[i].numpy() for i in range(len(ypreds_test))]
        acc = batch_acc(ypreds_test_numpy, y_test)
        return total_loss, total_acc, acc, ypreds_test_numpy, prune_information

    def process(self, arcs, inputs_bounds, weights_bounds):
        if self.give_up_rest:
            A = 0
            total_loss, total_acc, acc, ypre_test = 0, 0, 0, 0
        else:
            work_queue = []
            total_loss, total_acc, acc, ypre_test, prune_information = [], [], [], [], []
            for i in range(0, len(arcs)):
                work_queue.append([arcs[i], inputs_bounds[i], weights_bounds[i]])
            # result = self.train_circuit(work_queue[0])
            pool = Pool(processes=self.parallel)
            result = pool.map(self.train_circuit, work_queue)
            pool.close()
            pool.join()

            # Call garbage collector to free memory
            del work_queue
            gc.collect()

            for part in result:
                total_loss.append(part[0])
                total_acc.append(part[1])
                acc.append(part[2])
                ypre_test.append(part[3])
                prune_information.append(part[4])
        return total_loss, total_acc, acc, ypre_test, prune_information


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--task', type=str, default='Classification', help='see task_congigs.py for available tasks')
    parser.add_argument('--task_result', type=str, default='fmnist_4', help='')
    parser.add_argument('--search_space', type=str, default='block', help='search space')
    parser.add_argument('--method', type=str, default='MoE_QAS', help='search method')
    parser.add_argument('--qubit', type=int, default=4, help='')

    parser.add_argument("--train_N_pretraining", type=int, default=10000, help="")
    parser.add_argument('--num_initial_training_set', type=int, default=100, help='')

    #  trainer setting
    parser.add_argument('--batch_size', default=200, type=int, help='batch size for training')
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train the generated circuits for QNN")
    parser.add_argument("--iteration", type=int, default=2000, help="number of epochs to train the generated circuits for VQE")
    parser.add_argument("--parallel", type=int, default=10, help="multiprocessing training circuit")
    parser.add_argument("--train_lr", type=float, default=0.01, help="learning rate for training")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Early stopping parameters")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping parameters")
    parser.add_argument("--prune_check_num", type=int, default=40, help="")
    parser.add_argument("--softmax_flag", type=bool, default=False, help="")

    parser.add_argument("--noise", type=bool, default=True, help="whether to consider noise")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01, help="magnitude of the two-qubit depolarization noise")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001, help="magnitude of depolarization noise for a single qubit")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="bit-flip noise magnitude")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    '''Loading task data'''
    train_data = TensorFlowDataset(args.task_result, 'angle', 1, reshape_labels=True, file_type='txt')
    test_data = TensorFlowDataset(args.task_result, 'angle', 1, False, reshape_labels=True, file_type='txt')
    x_train = train_data.x_data
    y_train = train_data.y_data
    x_test = test_data.x_data
    y_test = test_data.y_data
    y_train_one_hot = get_one_hot(y_train)
    y_test_one_hot = get_one_hot(y_test)
    data_class = [x_train, y_train_one_hot, x_test, y_test_one_hot]

    qubit = args.qubit
    noise_param = None
    if args.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': args.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': args.single_qubit_depolarizing_p,
                       'bit_flip_p': args.bit_flip_p}

    """optimize circuit"""
    trainer_prune = ClassifyTrainerNew_prune(args, qubit, data_class, noise_param=noise_param)
    trainer = ClassifyTrainerNew(args, configs[args.task_result], data_class, noise_param=noise_param)

    cir = []
    inputs_bounds = []
    weights_bounds = []
    acc_list = []
    for i in range(5):
        path = f'../result/{args.task_result}/{args.method}/{args.search_space}/{args.train_N_pretraining}_{args.num_initial_training_set}/data/{i}/'
        cir.append(utils.load_pkl(path + f'list_cir_eval.pkl')[0])
        inputs_bounds.append(utils.load_pkl(path + f'inputs_bounds_eval.pkl')[0])
        weights_bounds.append(utils.load_pkl(path + f'weights_bounds_eval.pkl')[0])

        total_loss_eval, total_acc_eval, acc_eval, ypre_eval, prune_information = trainer_prune.process([cir[i]], [inputs_bounds[i]], [weights_bounds[i]])
        acc_list.append(acc_eval)
        utils.save_pkl(acc_eval, path + f'acc_eval_prune.pkl')
        utils.save_pkl(prune_information, path + f'prune_information.pkl')
    print(f'---------Result--------')
    print(f'Average top_1: {np.average(acc_list)}')
    print('------------------------')


