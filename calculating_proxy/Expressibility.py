import os
import numpy as np
from tqdm import tqdm
import tensorcircuit as tc
import tensorflow as tf
from scipy import stats
from utils import load_pkl, save_pkl
import random
import argparse
from multiprocessing import Pool
import time
tc.set_dtype("complex128")
tc.set_backend("tensorflow")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def data_divide(list_cir, inputs_bounds, weights_bounds, arg):
    np.random.seed(arg.seed)
    data = list(zip(list_cir, inputs_bounds, weights_bounds))
    np.random.shuffle(data)
    data_train = data[:arg.train_N]
    data_test = data[arg.train_N:]
    return data_train, data_test

class ExpressibilityCalculator:
    def __init__(self, num_random_initial, qubits, n_cir_parallel, noise_param, gaussian_kernel_sigma, seed) :
        self.pi_2 = 2 * np.pi
        self.num_random_initial = num_random_initial
        self.n_qubit = qubits
        self.n_cir_parallel = n_cir_parallel
        self.noise = False
        self.seed = seed
        self.gaussian_kernel_sigma = gaussian_kernel_sigma
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

    def quantum_circuit(self, structure, param):
        """
        :param param: Circuit parameters
        :param structure: Circuit
        :return: The final quantum state (statevector or density matrix, depending on noise model)
        """
        if self.noise:
            # print('noise: ', self.noise)
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            c = tc.DMCircuit(self.n_qubit)
            for i, gate in enumerate(structure):
                if gate.name == "CNOT":
                    c.cnot(gate.act_on[0], gate.act_on[1])
                    c.general_kraus(self.two_qubit_dep_channel, gate.act_on[0], gate.act_on[1])
                elif gate.name == "Ry":
                    c.ry(gate.act_on[0], theta=param[i][0])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                elif gate.name == "Rz":
                    c.rz(gate.act_on[0], theta=param[i][0])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                elif gate.name == "Rx":
                    c.rx(gate.act_on[0], theta=param[i][0])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                else:
                    print("invalid gate!")
                    exit(0)
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)
        else:
            c = tc.Circuit(self.n_qubit)
            for i, gate in enumerate(structure):
                if gate.name == "CNOT":
                    c.cnot(gate.act_on[0], gate.act_on[1])
                elif gate.name == "Ry":
                    c.ry(gate.act_on[0], theta=param[i][0])
                elif gate.name == "Rz":
                    c.rz(gate.act_on[0], theta=param[i][0])
                elif gate.name == "Rx":
                    c.rx(gate.act_on[0], theta=param[i][0])
                else:
                    print("invalid gate!")
                    exit(0)
        st = c.state()
        return st

    def get_parallel(self):
        parallel = tc.backend.vmap(self.quantum_circuit, vectorized_argnums=1)
        parallel = tc.backend.jit(parallel, static_argnums=(0))
        return parallel

    def train_circuit(self, circuit):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        random.seed(self.seed)
        parallel = self.get_parallel()
        par = np.random.uniform(0, 1, (self.num_random_initial, len(circuit), 1)) * self.pi_2
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )
        output_states = parallel(circuit, param)
        return output_states.numpy()

    def process(self, circuits):
        pool = Pool(processes=self.n_cir_parallel)
        quantum_state = pool.map(self.train_circuit, circuits)
        pool.close()
        pool.join()
        return quantum_state

    def fidelity_calculator(self, circuits):
        fidelities = []
        output_states = self.process(circuits)
        for i in range(len(output_states)):
            output_states1 = output_states[i][0:int(self.num_random_initial / 2)]
            output_states2 = output_states[i][int(self.num_random_initial / 2):]
            fidelity = (output_states1 * output_states2.conjugate()).sum(-1)
            fidelity = np.power(np.absolute(fidelity), 2)
            fidelities.append(fidelity)
        return fidelities

    def compute_expressibility_KL(self, circuits):
        N = self.n_qubit
        points = 100
        space = 1 / points
        x = [space * (i + 1) for i in range(-1, points)]
        haar_points = []
        for i in range(1, len(x)):
            temp1 = -1 * np.power((1 - x[i]), np.power(2, N) - 1)
            temp0 = -1 * np.power((1 - x[i - 1]), np.power(2, N) - 1)
            haar_points.append(temp1 - temp0)
        haar_points = np.array(haar_points)
        fidelities = self.fidelity_calculator(circuits)
        expressivity = []
        for inner in tqdm(fidelities, desc='Computing expressivity'):
            bin_index = np.floor(inner * points).astype(int)
            num = []
            for i in range(0, points):
                num.append(len(bin_index[bin_index == i]))
            num = np.array(num) / sum(num)
            output = stats.entropy(num, haar_points)
            expressivity.append(output)
        expressivity = np.array(expressivity)
        expressivity = -1 * expressivity
        return expressivity

    def get_expressibility(self, circuits):
        expressibility = self.compute_expressibility_KL(circuits)
        return expressibility

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--parallel", type=int, default=10, help="parallel processing")
    parser.add_argument("--proxy", type=str, default='EX', help="proxy file")
    parser.add_argument("--task", type=str, default='Classification_re', help="task")
    parser.add_argument('--search_space', type=str, default='gatewise', help='')
    parser.add_argument("--num_random_initial", type=int, default=2000, help="number of random initial for fidelities calcualtion")
    parser.add_argument("--qubits", type=int, default=4, help="qubit")
    parser.add_argument("--train_N", type=int, default=1000, help="")

    parser.add_argument("--noise", type=int, default=False, help="noise")
    parser.add_argument("--two_qubit_channel_depolarizing_p", type=float, default=0.01, help="two_qubit_noise")
    parser.add_argument("--single_qubit_channel_depolarizing_p", type=float, default=0.001, help="single_qubit_noise")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="bit_flip_noise")
    parser.add_argument("--haarPoints", type=int, default=100, help="KL_haar Points")
    parser.add_argument("--gaussian_kernel_sigma", type=float, default=0.01, help="MMD_gaussian_kernel_sigma")
    parser.add_argument('--pretraining', type=int, default=0, help='')
    args = parser.parse_args()
    start_time = time.time()

    noise_param = None
    if args.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': args.two_qubit_channel_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': args.single_qubit_channel_depolarizing_p,
                       'bit_flip_p': args.bit_flip_p}
    if args.pretraining == 0:
        list_cir = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/list_cir_{args.search_space}_seed{args.seed}/list_arc_{args.search_space}_seed{args.seed}.pkl')
        input_bounds = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/list_cir_{args.search_space}_seed{args.seed}/inputs_bounds_{args.search_space}_seed{args.seed}.pkl')
        weights_bounds = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/list_cir_{args.search_space}_seed{args.seed}/weights_bounds_{args.search_space}_seed{args.seed}.pkl')
        save_path = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/EX/'
        data_train, data_test = data_divide(list_cir, input_bounds, weights_bounds, args)
        list_cir_train, inputs_bounds_train, weights_bounds_train = zip(*data_train)
        list_cir_test, inputs_bounds_test, weights_bounds_test = zip(*data_test)
        list_cir = list_cir_test
        input_bounds = inputs_bounds_test
        weights_bounds = weights_bounds_test
    else:
        list_cir = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/list_cir_{args.search_space}_seed{args.seed}_pretrained/list_arc_{args.search_space}_seed{args.seed}_pretrained.pkl')
        input_bounds = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/list_cir_{args.search_space}_seed{args.seed}_pretrained/inputs_bounds_{args.search_space}_seed{args.seed}_pretrained.pkl')
        weights_bounds = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/list_cir_{args.search_space}_seed{args.seed}_pretrained/weights_bounds_{args.search_space}_seed{args.seed}_pretrained.pkl')
        save_path = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/EX/'

    Ex = ExpressibilityCalculator(args.num_random_initial, args.qubits, args.parallel, noise_param, args.gaussian_kernel_sigma, args.seed)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    expressibility = Ex.get_expressibility(list_cir)
    save_pkl(expressibility, save_path + f'expressibility.pkl')
    end_time = time.time()
    print(f'run time:{end_time - start_time}s')
    print('------------------------')
