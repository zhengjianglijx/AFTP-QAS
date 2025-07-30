import argparse
import os
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
import utils
from tqdm import tqdm
from multiprocessing import Pool
import matplotlib.pyplot as plt
import torch

from task_configs import configs

tc.set_dtype("complex128")
tc.set_backend("tensorflow")

class VqeTrainerNew_prune:
    def __init__(self, arg, n_cir_parallel, n_runs, max_iteration, n_qubit, hamiltonian, noise_param=None):

        self.K = tc.set_backend("tensorflow")
        self.n_qubit = n_qubit
        self.max_iteration = max_iteration
        self.n_cir_parallel = n_cir_parallel
        self.n_runs = n_runs
        self.prune_check_num = arg.prune_check_num
        self.softmax_flag = arg.softmax_flag
        self.hamiltonian_ = hamiltonian
        self.lattice = tc.templates.graphs.Line1D(self.n_qubit, pbc=self.hamiltonian_['pbc'])
        self.h = tc.quantum.heisenberg_hamiltonian(self.lattice, hzz=self.hamiltonian_['hzz'],
                                                   hxx=self.hamiltonian_['hxx'], hyy=self.hamiltonian_['hyy'],
                                                   hx=self.hamiltonian_['hx'], hy=self.hamiltonian_['hy'],
                                                   hz=self.hamiltonian_['hz'], sparse=self.hamiltonian_['sparse'])
        self.give_up_rest = False
        self.solution = None

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
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.two_qubit_channel_depolarizing_p/15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.single_qubit_channel_depolarizing_p/3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

    def compute_energy(self, param, structure, param_visit):
        """
        :param param:
        :param structure:
        :return:
        """
        if self.noise:
            # print('noise: ', self.noise)
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            c = tc.DMCircuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
                if gate.name == "CNOT":
                    c.cnot(gate.act_on[0], gate.act_on[1])
                    c.general_kraus(self.two_qubit_dep_channel, gate.act_on[0], gate.act_on[1])
                elif gate.name == "Ry":
                    if param_visit[param_index]:
                        c.ry(gate.act_on[0], theta=param[param_index][0])
                        c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                    param_index += 1
                elif gate.name == "Rz":
                    if param_visit[param_index]:
                        c.rz(gate.act_on[0], theta=param[param_index][0])
                        c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                    param_index += 1
                elif gate.name == "Rx":
                    if param_visit[param_index]:
                        c.rx(gate.act_on[0], theta=param[param_index][0])
                        c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                    param_index += 1
                else:
                    print("invalid gate!")
                    exit(0)
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            st = c.state()
            x = tf.matmul(st, self.h)
            e = tf.linalg.trace(x)
            e = self.K.real(e)

        else:
            c = tc.Circuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
                if gate.name == "CNOT":
                    c.cnot(gate.act_on[0], gate.act_on[1])
                elif gate.name == "Ry":
                    c.ry(gate.act_on[0], theta=param[param_index][0])
                    param_index += 1
                elif gate.name == "Rz":
                    c.rz(gate.act_on[0], theta=param[param_index][0])
                    param_index += 1
                elif gate.name == "Rx":
                    c.rx(gate.act_on[0], theta=param[param_index][0])
                    param_index += 1
                else:
                    print("invalid gate!")
                    exit(0)
            e = tc.templates.measurements.operator_expectation(c, self.h)

        return e

    def depth_count(self, cir, qubit):
        res = [0] * qubit
        # count = 0
        for gate in cir:
            if gate.qubits > 1:
                # num_two_qubit_gates += 1
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

    def get_param_num(self, cir):
        param_num = 0
        param_index = []
        for i in range(len(cir)):
            if cir[i].para_gate:
                param_num += 1
                param_index.append(i)
        return param_num, param_index

    def build_saliency(self, grad_buffer, thresh_vec, param_visit):
        # gbuf = grad_buffer.flatten()
        gbuf = grad_buffer
        prune_idx = []
        for i, g in enumerate(gbuf):
            if g < thresh_vec[i] and param_visit[i]:
                prune_idx.append(i)
        return prune_idx

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x))

    def train_circuit(self, circuit_and_seed):
        single_circuit = circuit_and_seed[0]
        seed = circuit_and_seed[1]
        np.random.seed(seed)
        tf.random.set_seed(seed)
        param_num, param_gate_index = self.get_param_num(single_circuit)
        # trainer = tc.backend.jit(tc.backend.value_and_grad(self.compute_energy, argnums=0))
        trainer = tc.backend.value_and_grad(self.compute_energy, argnums=0)
        L = self.depth_count(single_circuit, self.n_qubit)
        par = np.random.normal(loc=0, scale=1/(8*(L+2)), size=param_num)
        par = par.reshape((param_num, 1))
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )
        opt = tf.keras.optimizers.Adam(0.05)
        param_visit = np.ones(param_num, dtype=bool)  # 初始化为全 True
        e_last = 1000
        energy_epoch = []
        grad_list = []
        prune_index = []
        prune_information = []
        tau = [1/param_num] * param_num
        e, grad = trainer(param, single_circuit, param_visit)
        gbuffer = tf.squeeze(grad)
        grad_list.append(tf.squeeze(grad))
        energy_epoch.append(e.numpy())
        for i in range(0, self.max_iteration):
            if self.softmax_flag:
                tau = tau - tau * tf.nn.softmax(grad_list[-1])
            else:
                tau = tau - tau * abs(grad_list[-1])
            opt.apply_gradients([(grad, param)])
            e, grad = trainer(param, single_circuit, param_visit)
            grad_list.append(tf.squeeze(grad))
            energy_epoch.append(e.numpy())
            gbuffer = gbuffer + abs(grad_list[-2] - grad_list[-1])
            if i!= 0 and (i + 1) % self.prune_check_num == 0:
                idx = self.build_saliency(gbuffer.numpy(), tau.numpy(), param_visit)
                # print(f"Dropping {idx} for threshold {tau}")
                param_visit[idx] = False
                prune_index.extend(idx)
                gbuffer = tf.squeeze(grad)
                prune_information.append({'Step':i,'Energy':e,'Prune':prune_index, 'param_gate_index':param_gate_index})
                distance = abs(e_last - e.numpy())
                if distance < 0.0001:
                    # print(distance.max())
                    break
                else:
                    e_last = e.numpy()
        return e.numpy(), param.numpy(), energy_epoch, prune_information

    def process(self, arcs):

        if self.give_up_rest:
            A = 0
        else:
            work_queue = []
            for i in range(0, len(arcs)):
                work_queue.extend([[arcs[i], j] for j in range(0, self.n_runs)])
            # result = self.train_circuit(work_queue[0])
            pool = Pool(processes=self.n_cir_parallel)
            result = pool.map(self.train_circuit, work_queue)
            pool.close()
            pool.join()

            energy, param, energy_epoch, prune_information = [], [], [], []
            for part in result:
                energy.append(part[0])
                param.append(part[1])
                energy_epoch.append(part[2])
                prune_information.append(part[3])

        return energy, param, energy_epoch, prune_information



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--task', type=str, default='Classification', help='see task_congigs.py for available tasks')
    parser.add_argument('--task_result', type=str, default='TFIM', help='')
    parser.add_argument('--search_space', type=str, default='block', help='search space')
    parser.add_argument('--method', type=str, default='MoE_QAS', help='search method')
    parser.add_argument('--qubit', type=int, default=7, help='')

    parser.add_argument("--train_N_pretraining", type=int, default=5000, help="")
    parser.add_argument('--num_initial_training_set', type=int, default=100, help='')

    #  trainer setting
    parser.add_argument("--run", type=int, default=10, help="number of repetitions per line")
    parser.add_argument('--batch_size', default=200, type=int, help='batch size for training')
    parser.add_argument("--num_epochs", type=int, default=100,help="number of epochs to train the generated circuits for QNN")
    parser.add_argument("--iteration", type=int, default=2000, help="number of epochs to train the generated circuits for VQE")
    parser.add_argument("--parallel", type=int, default=10, help="multiprocessing training circuit")
    parser.add_argument("--train_lr", type=float, default=0.01, help="learning rate for training")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Early stopping parameters")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping parameters")
    parser.add_argument("--prune_check_num", type=int, default=300, help="")
    parser.add_argument("--softmax_flag", type=bool, default=False, help="")

    parser.add_argument("--noise", type=bool, default=True, help="whether to consider noise")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01, help="magnitude of the two-qubit depolarization noise")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001, help="magnitude of depolarization noise for a single qubit")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="bit-flip noise magnitude")
    args = parser.parse_args()
    cfgs = configs[args.task_result]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    qubit = args.seed
    noise_param = None
    if args.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': args.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': args.single_qubit_depolarizing_p,
                       'bit_flip_p': args.bit_flip_p}

    hamiltonian = configs[args.task_result]['Hamiltonian']
    trainer_prune = VqeTrainerNew_prune(arg=args, n_cir_parallel=args.parallel, n_runs=args.run, max_iteration=args.iteration, n_qubit=qubit, hamiltonian=hamiltonian, noise_param=noise_param)
    cir = []
    energy_list = []
    for i in range(5):
        path = f'../result/{args.task_result}/{args.method}/{args.search_space}/{args.train_N_pretraining}_{args.num_initial_training_set}/data/{i}/'
        cir.append(utils.load_pkl(path + f'list_cir_eval.pkl')[0])
        energy, param, energy_epoch, prune_information = trainer_prune.process([cir[i]])
        energy_list.append(energy)
        utils.save_pkl(energy, path + f'energy_prune_prune.pkl')
        utils.save_pkl(prune_information, path + f'prune_information.pkl')
    print(f'---------Result--------')
    print(f'Average top_1: {np.average(energy_list)}')
    print('------------------------')


