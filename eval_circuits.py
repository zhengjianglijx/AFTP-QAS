import utils
import os
import argparse
from task_configs import configs
import time
import numpy as np
import torch
from data_process_T import TensorFlowDataset
from optimize_circuits.classify_task_paralell import ClassifyTrainerNew
from optimize_circuits.vqe_task_paralell import VqeTrainerNew
import tensorcircuit as tc
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

def eval_Single_proxy_QNN(data_class, cfg, arg, method):
    seed = [0, 1, 2, 3, 4]
    index_eval = []
    list_cir_eval = []
    inputs_bounds_eval = []
    weights_bounds_eval = []
    for c in seed:
        save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/data/{c}/'
        index_eval.extend(utils.load_pkl(f'{save_result_data}/index_eval.pkl'))
        list_cir_eval.extend(utils.load_pkl(f'{save_result_data}/list_cir_eval.pkl'))
        inputs_bounds_eval.extend(utils.load_pkl(f'{save_result_data}/inputs_bounds_eval.pkl'))
        weights_bounds_eval.extend(utils.load_pkl(f'{save_result_data}/weights_bounds_eval.pkl'))

    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    trainer = ClassifyTrainerNew(arg, cfg, data_class, noise_param=noise_param)
    total_loss_eval, total_acc_eval, acc_eval, ypre_eval = trainer.process(list_cir_eval, inputs_bounds_eval, weights_bounds_eval)
    utils.save_pkl(acc_eval, f'result/{arg.task_result}/{method}/{arg.search_space}/acc_eval.pkl')
    expected_shape = (len(seed), 5, arg.query_num)
    acc_eval = np.array(acc_eval)
    acc_eval_reshaped = np.reshape(acc_eval, expected_shape)
    for k in range(len(seed)):
        save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/data/{seed[k]}/'
        utils.save_pkl(acc_eval_reshaped[k], f'{save_result_data}/acc_eval.pkl')
        utils.save_pkl(acc_eval_reshaped[k, :, 0], f'{save_result_data}/top_1.pkl')
    print(f'---------Result--------')
    print(f'Accuracy: {acc_eval_reshaped}')
    print(f'Average top_1: {np.average(acc_eval_reshaped[:, :, 0], 0)}')
    print('------------------------')

def eval_MoE_QNN(data_class, cfg, arg, method):
    num_initial_training_set = [arg.num_initial_training_set]
    seed = [0, 1, 2, 3, 4]
    index_eval = []
    list_cir_eval = []
    inputs_bounds_eval = []
    weights_bounds_eval = []
    for a in num_initial_training_set:
        for c in seed:
            save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/{arg.train_N_pretraining}_{a}/data/{c}/'
            index_eval.extend(utils.load_pkl(f'{save_result_data}/index_eval.pkl'))
            list_cir_eval.extend(utils.load_pkl(f'{save_result_data}/list_cir_eval.pkl'))
            inputs_bounds_eval.extend(utils.load_pkl(f'{save_result_data}/inputs_bounds_eval.pkl'))
            weights_bounds_eval.extend(utils.load_pkl(f'{save_result_data}/weights_bounds_eval.pkl'))
    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    trainer = ClassifyTrainerNew(arg, cfg, data_class, noise_param=noise_param)
    total_loss_eval, total_acc_eval, acc_eval, ypre_eval = trainer.process(list_cir_eval, inputs_bounds_eval, weights_bounds_eval)
    utils.save_pkl(acc_eval, f'result/{arg.task_result}/{method}/{arg.search_space}/acc_eval_{arg.train_N_pretraining}_{num_initial_training_set[0]}.pkl')
    expected_shape = (len(num_initial_training_set), len(seed), arg.query_num)
    acc_eval = np.array(acc_eval)
    acc_eval_reshaped = np.reshape(acc_eval, expected_shape)  # Reshape the data to the specified shape

    for i in range(len(num_initial_training_set)):
        for k in range(len(seed)):
            save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/{arg.train_N_pretraining}_{num_initial_training_set[i]}/data/{seed[k]}/'
            utils.save_pkl(acc_eval_reshaped[i][k], f'{save_result_data}/acc_eval.pkl')
            utils.save_pkl(acc_eval_reshaped[i][k][0], f'{save_result_data}/top_1.pkl')
        print(f'---------Result--------')
        print(f'Accuracy: {acc_eval_reshaped[i]}')
        print(f'Average top_1: {np.average(acc_eval_reshaped[i][:, 0])}')
        print('------------------------')

def eval_MoE_removal_QNN(data_class, cfg, arg, method):
    num_initial_training_set = [arg.num_initial_training_set]
    seed = [0, 1, 2, 3, 4]
    expert_to_remove = [0, 1, 2, 3, 4, 5]
    index_eval = []
    list_cir_eval = []
    inputs_bounds_eval = []
    weights_bounds_eval = []
    for a in num_initial_training_set:
        for b in expert_to_remove:
            for c in seed:
                save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/{arg.train_N_pretraining}_{a}/{b}/data/{c}/'
                index_eval.extend(utils.load_pkl(f'{save_result_data}/index_eval.pkl'))
                list_cir_eval.extend(utils.load_pkl(f'{save_result_data}/list_cir_eval.pkl'))
                inputs_bounds_eval.extend(utils.load_pkl(f'{save_result_data}/inputs_bounds_eval.pkl'))
                weights_bounds_eval.extend(utils.load_pkl(f'{save_result_data}/weights_bounds_eval.pkl'))
    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    trainer = ClassifyTrainerNew(arg, cfg, data_class, noise_param=noise_param)
    total_loss_eval, total_acc_eval, acc_eval, ypre_eval = trainer.process(list_cir_eval, inputs_bounds_eval, weights_bounds_eval)
    utils.save_pkl(acc_eval, f'result/{arg.task_result}/{method}/{arg.search_space}/acc_eval_{arg.train_N_pretraining}_{num_initial_training_set[0]}.pkl')
    expected_shape = (len(num_initial_training_set), len(expert_to_remove), len(seed), arg.query_num)
    acc_eval = np.array(acc_eval)
    acc_eval_reshaped = np.reshape(acc_eval, expected_shape)  # Reshape the data to the specified shape

    for i in range(len(num_initial_training_set)):
        for j in range(len(expert_to_remove)):
            for k in range(len(seed)):
                save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/{arg.train_N_pretraining}_{num_initial_training_set[i]}/{expert_to_remove[j]}/data/{seed[k]}/'
                utils.save_pkl(acc_eval_reshaped[i][j][k], f'{save_result_data}/acc_eval.pkl')
                utils.save_pkl(acc_eval_reshaped[i][j][k][0], f'{save_result_data}/top_1.pkl')
            print(f'---------Result--------')
            print(f'Accuracy: {acc_eval_reshaped[i][j]}')
            print(f'Average top_1: {np.average(acc_eval_reshaped[i][j][:, 0])}')
            print('------------------------')

def eval_Liner_QAS_QNN(data_class, cfg, arg, method):
    seed = [0, 1, 2, 3, 4]
    index_eval = []
    list_cir_eval = []
    inputs_bounds_eval = []
    weights_bounds_eval = []
    for c in seed:
        save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/data/{c}/'
        index_eval.extend(utils.load_pkl(f'{save_result_data}/index_eval.pkl'))
        list_cir_eval.extend(utils.load_pkl(f'{save_result_data}/list_cir_eval.pkl'))
        inputs_bounds_eval.extend(utils.load_pkl(f'{save_result_data}/inputs_bounds_eval.pkl'))
        weights_bounds_eval.extend(utils.load_pkl(f'{save_result_data}/weights_bounds_eval.pkl'))

    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    trainer = ClassifyTrainerNew(arg, cfg, data_class, noise_param=noise_param)
    total_loss_eval, total_acc_eval, acc_eval, ypre_eval = trainer.process(list_cir_eval, inputs_bounds_eval, weights_bounds_eval)
    utils.save_pkl(acc_eval, f'result/{arg.task_result}/{method}/{arg.search_space}/acc_eval.pkl')
    expected_shape = (len(seed), arg.query_num)
    acc_eval = np.array(acc_eval)
    acc_eval_reshaped = np.reshape(acc_eval, expected_shape)
    for k in range(len(seed)):
        save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/data/{seed[k]}/'
        utils.save_pkl(acc_eval_reshaped[k], f'{save_result_data}/acc_eval.pkl')
        utils.save_pkl(acc_eval_reshaped[k][0], f'{save_result_data}/top_1.pkl')
    print(f'---------Result--------')
    print(f'Accuracy: {acc_eval_reshaped}')
    print(f'Average top_1: {np.average(acc_eval_reshaped[:, 0])}')
    print('------------------------')

def eval_Single_proxy_VQE(cfg, arg, method):
    seed = [0, 1, 2, 3, 4]
    index_eval = []
    list_cir_eval = []
    for c in seed:
        save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/data/{c}/'
        index_eval.extend(utils.load_pkl(f'{save_result_data}/index_eval.pkl'))
        list_cir_eval.extend(utils.load_pkl(f'{save_result_data}/list_cir_eval.pkl'))

    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    trainer = VqeTrainerNew(task_configs=cfg, n_cir_parallel=arg.parallel, n_runs=arg.run, max_iteration=arg.iteration, noise_param=noise_param)
    energy_result, param_result, energy_epoch = trainer.process(list_cir_eval)
    utils.save_pkl(energy_result, f'result/{arg.task_result}/{method}/{arg.search_space}/energy_eval.pkl')
    expected_shape = (len(seed), 5, arg.query_num)
    energy_result = np.array(energy_result)
    energy_eval_reshaped = np.reshape(energy_result, expected_shape)
    for k in range(len(seed)):
        save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/data/{seed[k]}/'
        utils.save_pkl(energy_eval_reshaped[k], f'{save_result_data}/energy.pkl')
        utils.save_pkl(energy_eval_reshaped[k, :, 0], f'{save_result_data}/top_1.pkl')
    print(f'---------Result--------')
    print(f'Energy: {energy_eval_reshaped}')
    print(f'Average top_1: {np.average(energy_eval_reshaped[:, :, 0], 0)}')
    print('------------------------')

def eval_MoE_VQE(cfg, arg, method):
    num_initial_training_set = [arg.num_initial_training_set]
    seed = [0, 1, 2, 3, 4]
    index_eval = []
    list_cir_eval = []
    for a in num_initial_training_set:
        for c in seed:
            save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/{arg.train_N_pretraining}_{a}/data/{c}/'
            index_eval.extend(utils.load_pkl(f'{save_result_data}/index_eval.pkl'))
            list_cir_eval.extend(utils.load_pkl(f'{save_result_data}/list_cir_eval.pkl'))

    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    trainer = VqeTrainerNew(task_configs=cfg, n_cir_parallel=arg.parallel, n_runs=arg.run, max_iteration=arg.iteration, noise_param=noise_param)
    energy_result, param_result, energy_epoch = trainer.process(list_cir_eval)
    utils.save_pkl(energy_result, f'result/{arg.task_result}/{method}/{arg.search_space}/energy_{arg.train_N_pretraining}_{num_initial_training_set[0]}.pkl')
    expected_shape = (len(num_initial_training_set), len(seed), arg.query_num)
    energy_result = np.array(energy_result)
    energy_eval_reshaped = np.reshape(energy_result, expected_shape)

    for i in range(len(num_initial_training_set)):
        for k in range(len(seed)):
            save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/{arg.train_N_pretraining}_{num_initial_training_set[i]}/data/{seed[k]}/'
            utils.save_pkl(energy_eval_reshaped[i][k], f'{save_result_data}/energy.pkl')
            utils.save_pkl(energy_eval_reshaped[i][k][0], f'{save_result_data}/top_1.pkl')
        print(f'---------Result--------')
        print(f'Energy: {energy_eval_reshaped[i]}')
        print(f'Average top_1: {np.average(energy_eval_reshaped[i][:, 0])}')
        print('------------------------')

def eval_MoE_removal_VQE(cfg, arg, method):
    num_initial_training_set = [arg.num_initial_training_set]
    seed = [0, 1, 2, 3, 4]
    expert_to_remove = [0, 1, 2, 3, 4, 5]
    index_eval = []
    list_cir_eval = []
    for a in num_initial_training_set:
        for b in expert_to_remove:
            for c in seed:
                save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/{arg.train_N_pretraining}_{a}/{b}/data/{c}/'
                index_eval.extend(utils.load_pkl(f'{save_result_data}/index_eval.pkl'))
                list_cir_eval.extend(utils.load_pkl(f'{save_result_data}/list_cir_eval.pkl'))

    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    trainer = VqeTrainerNew(task_configs=cfg, n_cir_parallel=arg.parallel, n_runs=arg.run, max_iteration=arg.iteration, noise_param=noise_param)
    energy_result, param_result, energy_epoch = trainer.process(list_cir_eval)
    utils.save_pkl(energy_result, f'result/{arg.task_result}/{method}/{arg.search_space}/energy_{arg.train_N_pretraining}_{num_initial_training_set[0]}.pkl')
    expected_shape = (len(num_initial_training_set), len(expert_to_remove), len(seed), arg.query_num)
    energy_result = np.array(energy_result)
    energy_eval_reshaped = np.reshape(energy_result, expected_shape)

    for i in range(len(num_initial_training_set)):
        for j in range(len(expert_to_remove)):
            for k in range(len(seed)):
                save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/{arg.train_N_pretraining}_{num_initial_training_set[i]}/{expert_to_remove[j]}/data/{seed[k]}/'
                utils.save_pkl(energy_eval_reshaped[i][j][k], f'{save_result_data}/energy.pkl')
                utils.save_pkl(energy_eval_reshaped[i][j][k][0], f'{save_result_data}/top_1.pkl')
            print(f'---------Result--------')
            print(f'Energy: {energy_eval_reshaped[i][j]}')
            print(f'Average top_1: {np.average(energy_eval_reshaped[i][j][:, 0])}')
            print('------------------------')

def eval_Liner_QAS_VQE(cfg, arg, method):
    seed = [0, 1, 2, 3, 4]
    index_eval = []
    list_cir_eval = []
    for c in seed:
        save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/data/{c}/'
        index_eval.extend(utils.load_pkl(f'{save_result_data}/index_eval.pkl'))
        list_cir_eval.extend(utils.load_pkl(f'{save_result_data}/list_cir_eval.pkl'))

    noise_param = None
    if arg.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': arg.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': arg.single_qubit_depolarizing_p,
                       'bit_flip_p': arg.bit_flip_p}

    """optimize circuit"""
    trainer = VqeTrainerNew(task_configs=cfg, n_cir_parallel=arg.parallel, n_runs=arg.run, max_iteration=arg.iteration, noise_param=noise_param)
    energy_result, param_result, energy_epoch = trainer.process(list_cir_eval)
    utils.save_pkl(energy_result, f'result/{arg.task_result}/{method}/{arg.search_space}/energy.pkl')
    expected_shape = (len(seed), arg.query_num)
    energy_result = np.array(energy_result)
    energy_eval_reshaped = np.reshape(energy_result, expected_shape)

    for k in range(len(seed)):
        save_result_data = f'result/{arg.task_result}/{method}/{arg.search_space}/data/{seed[k]}/'
        utils.save_pkl(energy_eval_reshaped[k], f'{save_result_data}/energy.pkl')
        utils.save_pkl(energy_eval_reshaped[k][0], f'{save_result_data}/top_1.pkl')
    print(f'---------Result--------')
    print(f'Energy: {energy_eval_reshaped}')
    print(f'Average top_1: {np.average(energy_eval_reshaped[:, 0])}')
    print('------------------------')

def main(arg, cfg):
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

    if arg.task == 'Classification':
        train_data = TensorFlowDataset(arg.task_result, 'angle', cfg['num_data_reps'], reshape_labels=True, file_type=cfg['file_type'])
        test_data = TensorFlowDataset(arg.task_result, 'angle', cfg['num_data_reps'], False, reshape_labels=True, file_type=cfg['file_type'])
        x_train = train_data.x_data
        y_train = train_data.y_data
        x_test = test_data.x_data
        y_test = test_data.y_data
        y_train_one_hot = get_one_hot(y_train)
        y_test_one_hot = get_one_hot(y_test)
        data_class = [x_train, y_train_one_hot, x_test, y_test_one_hot]
        if arg.method == 'MoE_QAS':
            eval_MoE_QNN(data_class, cfg, arg, arg.method)
        if arg.method == 'Liner_QAS':
            eval_Liner_QAS_QNN(data_class, cfg, arg, arg.method)
        if arg.method == 'NonLiner_QAS':
            eval_Liner_QAS_QNN(data_class, cfg, arg, arg.method)
        if arg.method == 'Single_proxy':
            eval_Single_proxy_QNN(data_class, cfg, arg, arg.method)
        if arg.method == 'MoE_QAS_removal':
            eval_MoE_removal_QNN(data_class, cfg, arg, arg.method)
    else:
        if arg.method == 'MoE_QAS':
            eval_MoE_VQE(cfg, arg, arg.method)
        if arg.method == 'Liner_QAS':
            eval_Liner_QAS_VQE(cfg, arg, arg.method)
        if arg.method == 'NonLiner_QAS':
            eval_Liner_QAS_VQE(cfg, arg, arg.method)
        if arg.method == 'Single_proxy':
            eval_Single_proxy_VQE(cfg, arg, arg.method)
        if arg.method == 'MoE_QAS_removal':
            eval_MoE_removal_VQE(cfg, arg, arg.method)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--task', type=str, default='Classification', help='see congigs for available tasks')
    parser.add_argument('--search_space', type=str, default='block', help='')
    parser.add_argument('--task_result', type=str, default='fmnist_4', help='')
    parser.add_argument('--method', type=str, default='MoE_QAS', help='Search method')
    parser.add_argument('--qubit', type=int, default=4, help='')
    parser.add_argument('--num_initial_training_set', type=int, default=100, help='')
    parser.add_argument("--train_N_pretraining", type=int, default=10000, help="")
    parser.add_argument('--query_num', type=int, default=1, help='')

    #  trainer setting
    parser.add_argument("--run", type=int, default=10, help="The number of repetitions per line")
    parser.add_argument('--batch_size', default=200, type=int, help='batch size for training')
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train the generated circuits for QML")
    parser.add_argument("--iteration", type=int, default=2000, help="number of epochs to train the generated circuits for")
    parser.add_argument("--parallel", type=int, default=10, help="Parallel training circuit")
    parser.add_argument("--train_lr", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--min_delta", type=float, default=0.001, help="Early stopping parameters")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping parameters")

    parser.add_argument("--noise", type=bool, default=True, help="whether to consider noise")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01, help="magnitude of the two-qubit depolarization noise")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001, help="magnitude of depolarization noise for a single qubit")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="bit-flip noise magnitude")

    args = parser.parse_args()
    cfgs = configs[args.task_result]
    main(args, cfgs)