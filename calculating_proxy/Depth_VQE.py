import os.path
import numpy as np
import time
import networkx as nx
from utils import save_pkl, load_pkl
from multiprocessing import Pool
import argparse
from architecture_generator import ArchitectureGenerator

def data_divide(cir_list, arg):
    np.random.seed(arg.seed)
    data = cir_list
    np.random.shuffle(data)
    data_train = data[:arg.train_N]
    data_test = data[arg.train_N:]
    return data_train, data_test

def calculate_mean_path_length(work_queue):
    """
    Calculate the average path length from the first node to the last node in a single adjacency matrix.
    Parameters:
        adj_matrix: A single adjacency matrix
    Returns:
        The average path length of this adjacency matrix
    """
    adj_matrix = work_queue[0]
    index = work_queue[1]
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    all_paths = list(nx.all_simple_paths(G, source=0, target=len(adj_matrix) - 1))
    path_lengths = [len(path) - 1 for path in all_paths]
    mean_length = np.mean(path_lengths)
    print(f'The {index} circuits is computed')
    return mean_length

def depth(matrix_arc, save_path, arg):
    t_start = time.time()
    adj = [matrix_arc[i][0] for i in range(len(matrix_arc))]
    work_queue = []
    for i in range(len(adj)):
        work_queue.append([adj[i], i])
    pool = Pool(processes=arg.parallel)
    depth_num = pool.map(calculate_mean_path_length, work_queue)
    pool.close()
    pool.join()
    t_end = time.time()
    print(f'time_for_computing_depth:{t_end-t_start}s')
    depth_num = np.array(depth_num)
    save_pkl(depth_num, save_path + 'depth_number.pkl')
    return depth_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--proxy", type=str, default='Depth', help="proxy file")
    parser.add_argument("--task", type=str, default='VQE', help="task")
    parser.add_argument('--search_space', type=str, default='block', help='')
    parser.add_argument("--parallel", type=int, default=10, help="parallel processing")
    parser.add_argument("--qubits", type=int, default=7, help="qubit")
    parser.add_argument('--pretraining', type=int, default=0, help='')
    parser.add_argument("--train_N", type=int, default=1000, help="")
    args = parser.parse_args()

    if args.pretraining == 0:
        list_cir = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/list_cir_{args.search_space}_seed{args.seed}/list_arc_{args.search_space}_seed{args.seed}.pkl')
        save_path = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}/Depth/'
        data_train, data_test = data_divide(list_cir, args)
        list_cir = data_test
    else:
        list_cir = load_pkl(f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/list_cir_{args.search_space}_seed{args.seed}_pretrained/list_arc_{args.search_space}_seed{args.seed}_pretrained.pkl')
        save_path = f'../datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/Depth/'



    ag = ArchitectureGenerator(gate_pool=[{"name": "Ry", "qubits": 1, "para_gate": True},
                                          {"name": "Rz", "qubits": 1, "para_gate": True},
                                          {"name": "Rx", "qubits": 1, "para_gate": True},
                                          {"name": "CNOT", "qubits": 2, "para_gate": False}],
                               two_gate_num_range=[8, 14],
                               num_layers=-1,
                               num_qubits=args.qubits,
                               not_first=[{"name": "CNOT", "qubits": 2, "para_gate": False}],
                               start_with_u=True,
                               num_embed_gates=16,
                               )
    matrix_arc = ag.list_to_adj(list_cir)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    path_num = depth(matrix_arc, save_path, args)
    print('a')







