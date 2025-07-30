import os
import numpy as np
from tqdm import tqdm
import tensorcircuit as tc
import tensorflow as tf
from utils import load_pkl, save_pkl
import argparse
from multiprocessing import Pool
from qiskit.quantum_info import random_statevector
import time

tc.set_dtype("complex128")
tc.set_backend("tensorflow")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def data_divide(list_cir, arg):
    np.random.seed(arg.seed)
    data = list(zip(list_cir))
    np.random.shuffle(data)
    data_train = data[:arg.train_N]
    data_test = data[arg.train_N:]
    return data_train, data_test

class SnipCalculator:
    def __init__(self, arg, noise_param=None):
        self.n_qubit = arg.qubits
        self.parallel = arg.parallel
        self.num_random_initial = arg.num_random_initial
        self.seed = 0
        self.give_up_rest = False
        self.solution = None

        """ Noise-related parameter, don't care if noise is False. """
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

    def compute_energy(self, param, structure, random_vector):
        """
        :param: Circuit Parameters
        :structure: Circuit
        :random_vector: Using a random quantum state as the target state to compute the loss function
        :return: loss
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
                    c.ry(gate.act_on[0], theta=param[param_index])
                    param_index += 1
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                elif gate.name == "Rz":
                    c.rz(gate.act_on[0], theta=param[param_index])
                    param_index += 1
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                elif gate.name == "Rx":
                    c.rx(gate.act_on[0], theta=param[param_index])
                    param_index
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                else:
                    print("invalid gate!")
                    exit(0)
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            st = c.state()
        else:
            c = tc.Circuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
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
            st = c.state()
        random_vector_tensor = tf.convert_to_tensor(random_vector, dtype=tf.complex128)
        loss = -1 * tf.square(tf.abs(tf.reduce_sum(st * tf.math.conj(random_vector_tensor))))
        return loss

    def get_parallel(self):
        parallel = tc.backend.value_and_grad(self.compute_energy, argnums=(0,))
        parallel = tc.backend.jit(parallel, static_argnums=(1, 2))
        return parallel

    def get_param_num(self, cir):
        param_num = 0
        for i in range(len(cir)):
            if cir[i].para_gate:
                param_num += 1
        return param_num

    def train_circuit(self, work_queue):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        parallel = self.get_parallel()
        single_circuit, random_vector = work_queue[0], work_queue[1]
        param_num = self.get_param_num(single_circuit)
        par = np.random.uniform(0, 1, (self.num_random_initial, param_num)) * np.pi * 2
        grads = []
        params = []
        for i in range(self.num_random_initial):
            temp_param = tf.Variable(initial_value=tf.convert_to_tensor(par[i], dtype=getattr(tf, tc.rdtypestr)))
            loss, grad = parallel(temp_param, single_circuit, random_vector)
            grad = tf.squeeze(grad)
            grads.append(grad.numpy())
            params.append(temp_param.numpy())
        return grads, params

    def process(self, circuits, random_vector):
        work_queue = []
        for i in range(len(circuits)):
            work_queue.append([circuits[i], random_vector])

        pool = Pool(processes=self.parallel)
        results = pool.map(self.train_circuit, work_queue)
        pool.close()
        pool.join()
        grad = []
        param = []
        for part in results:
            grad.append(part[0])
            param.append(part[1])
        return grad, param

    def get_trainability(self, grad):
        trainability = []
        for i in range(self.num_random_initial):
            trainability.append(tf.norm(grad[i]).numpy())
        return trainability

    def get_snip_list(self, grad, param):
        valid_snip_list = []
        for i in range(self.num_random_initial):
            temp = []
            for j in range(len(grad[i])):
                temp.append(np.abs(grad[i][j] * param[i][j]))
            valid_snip_list.append(np.mean(temp))
        return valid_snip_list

    def Snip_calculator(self, circuits):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        snip = []
        random_vector = random_statevector(2 ** self.n_qubit).data
        grads, params = self.process(circuits, random_vector)

        for index in tqdm(range(0, len(circuits)), desc='Computing snip'):
            snip_list = self.get_snip_list(grads[index], params[index])
            snip.append(np.mean(snip_list))
        return snip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--proxy", type=str, default='Tr', help="proxy file")
    parser.add_argument("--task", type=str, default='VQE', help="task")
    parser.add_argument('--search_space', type=str, default='block', help='')
    parser.add_argument("--parallel", type=int, default=10, help="parallel processing")
    parser.add_argument("--num_random_initial", type=int, default=32, help="number of random initial for fidelities calcualtion")
    parser.add_argument("--qubits", type=int, default=7, help="qubit")

    parser.add_argument("--noise", type=int, default=False, help="noise")
    parser.add_argument("--two_qubit_channel_depolarizing_p", type=float, default=0.01, help="two_qubit_noise")
    parser.add_argument("--single_qubit_channel_depolarizing_p", type=float, default=0.001, help="single_qubit_noise")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="bit_flip_noise")
    parser.add_argument('--pretraining', type=int, default=0, help='')
    args = parser.parse_args()
    start_time = time.time()

    noise_param = None
    if args.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': 0.01,
                       'single_qubit_channel_depolarizing_p': 0.001,
                       'bit_flip_p': 0.01}

    '''Load data'''
    if args.pretraining == 0:
        list_cir = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/list_cir_{args.search_space}_seed{args.seed}/list_arc_{args.search_space}_seed{args.seed}.pkl')
        save_path_s = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/Snip_task_w/'
        data_train, data_test = data_divide(list_cir, args)
        list_cir_train = zip(*data_train)
        list_cir_test = zip(*data_test)
        list_cir = list_cir_test
    else:
        list_cir = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/list_cir_{args.search_space}_seed{args.seed}_pretrained/list_arc_{args.search_space}_seed{args.seed}_pretrained.pkl')
        save_path_s = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/Snip_task_w/'

    if not os.path.exists(save_path_s):
        os.makedirs(save_path_s)
    Tr = SnipCalculator(args, noise_param=noise_param)
    snip = Tr.Snip_calculator(list_cir)
    print('a')
    save_pkl(snip, save_path_s + f'snip.pkl')
    end_time = time.time()
    print(f'run time:{end_time - start_time}s')
    print('------------------------')
