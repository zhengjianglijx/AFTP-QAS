import os
import numpy as np
from tqdm import tqdm
import tensorcircuit as tc
import tensorflow as tf
from utils import load_pkl, save_pkl
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

class TrainabilityCalculator:
    def __init__(self, arg, noise_param=None):
        self.n_qubit = arg.qubits
        self.num_embed_gates = arg.num_embed_gates
        self.parallel = arg.parallel
        self.num_classes = arg.num_classes
        self.sel_samples_per_class = arg.num_samples_per_class
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


    def compute_energy(self, x, y, param, structure, embed_flags):
        """
        :x, y: where x is a randomly generated image feature vector and y is its associated class label
        :param: Circuit Parameters
        :structure: Circuit
        :random_vector: Using a random quantum state as the target state to compute the loss function
        :return: loss
        """
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
                        c.ry(gate.act_on[0], theta=param[param_index])
                        c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                        param_index += 1
                    elif gate.name == "Rz":
                        c.rz(gate.act_on[0], theta=param[param_index])
                        c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                        param_index += 1
                    elif gate.name == "Rx":
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

    def get_parallel(self):
        parallel = tc.backend.vectorized_value_and_grad(self.compute_energy, argnums=(2,),vectorized_argnums=(0, 1), has_aux=True)
        parallel = tc.backend.jit(parallel, static_argnums=(3, 4))
        return parallel

    def train_circuit(self, work_queue):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        parallel = self.get_parallel()
        embed_flags = []
        param_flags = []
        single_circuit, x, y, inputs_bounds, weights_bounds = work_queue[0], work_queue[1], work_queue[2], work_queue[3], work_queue[4]
        embed_flags, param_flags = self.generate_param_emb_index(single_circuit, inputs_bounds, weights_bounds, embed_flags, param_flags)

        par = np.random.uniform(0, 1, (self.num_random_initial, weights_bounds[-1])) * np.pi * 2
        grads = []
        params = []
        for i in range(self.num_random_initial):
            temp_param = tf.Variable(initial_value=tf.convert_to_tensor(par[i], dtype=getattr(tf, tc.rdtypestr)))
            (loss, ypreds), grad = parallel(x, y, temp_param, single_circuit, embed_flags)
            grad = tf.squeeze(grad)
            grads.append(grad.numpy())
            params.append(temp_param.numpy())
        return grads, params

    def process(self, circuits, sel_data_x, sel_data_y, inputs_bounds, weights_bounds):
        work_queue = []
        for i in range(len(circuits)):
            work_queue.append([circuits[i], sel_data_x, sel_data_y, inputs_bounds[i], weights_bounds[i]])

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


    def Trainability_calculator(self, circuits, inputs_bounds, weights_bounds):
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        trainabilities = []
        snip = []
        batch_size = self.num_classes * self.sel_samples_per_class
        x_train = np.random.uniform(0, 1, (batch_size, self.num_embed_gates)) * 2 * np.pi
        y = np.random.randint(low=0, high=self.num_classes, size=batch_size)
        y_train = np.eye(self.num_classes)[y]
        y_train = y_train.astype(np.float64)
        grads, params = self.process(circuits, x_train, y_train, inputs_bounds, weights_bounds)
        print('a')
        for index in tqdm(range(0, len(circuits)), desc='Computing trainability'):
            t = self.get_trainability(grads[index])
            trainabilities.append(np.mean(t))

        for index in tqdm(range(0, len(circuits)), desc='Computing snip'):
            snip_list = self.get_snip_list(grads[index], params[index])
            snip.append(np.mean(snip_list))
        return trainabilities, snip


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--proxy", type=str, default='Tr', help="proxy file")
    parser.add_argument("--task", type=str, default='Classification_re', help="task")
    parser.add_argument("--num_embed_gates", type=int, default=16, help=" ")
    parser.add_argument('--search_space', type=str, default='gatewise', help='')
    parser.add_argument("--parallel", type=int, default=10, help="parallel processing")
    parser.add_argument("--num_random_initial", type=int, default=32, help="number of random initial for fidelities calcualtion")
    parser.add_argument("--qubits", type=int, default=4, help="qubit")
    parser.add_argument("--train_N", type=int, default=1000, help="")

    parser.add_argument('--num_classes', type=int, default=4, help='number of samples to use per class in the dataset')
    parser.add_argument('--num_samples_per_class', type=int, default=16, help='number of samples to use per class in the dataset')
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
        input_bounds = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/list_cir_{args.search_space}_seed{args.seed}/inputs_bounds_{args.search_space}_seed{args.seed}.pkl')
        weights_bounds = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/list_cir_{args.search_space}_seed{args.seed}/weights_bounds_{args.search_space}_seed{args.seed}.pkl')
        save_path_t = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/Tr_task_w/'
        save_path_s = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/Snip_task_w/'
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
        save_path_t = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/Tr_task_w/'
        save_path_s = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/Snip_task_w/'

    if not os.path.exists(save_path_t):
        os.makedirs(save_path_t)

    if not os.path.exists(save_path_s):
        os.makedirs(save_path_s)

    Tr = TrainabilityCalculator(args, noise_param=noise_param)
    trainability, snip = Tr.Trainability_calculator(list_cir, input_bounds, weights_bounds)

    print('a')
    save_pkl(trainability, save_path_t + f'trainability.pkl')
    save_pkl(snip, save_path_s + f'snip.pkl')
    end_time = time.time()
    print(f'run time:{end_time - start_time}s')
    print('------------------------')
