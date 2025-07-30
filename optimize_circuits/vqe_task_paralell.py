import os
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
tc.set_dtype("complex128")

class VqeTrainerNew:
    def __init__(self, n_cir_parallel, n_runs, max_iteration, task_configs, noise_param=None):

        self.K = tc.set_backend("tensorflow")
        self.gate_pool = task_configs['gate_pool']
        self.n_qubit = task_configs['qubit']
        self.max_iteration = max_iteration
        self.n_cir_parallel = n_cir_parallel
        self.n_runs = n_runs
        self.hamiltonian_ = task_configs['Hamiltonian']
        self.lattice = tc.templates.graphs.Line1D(self.n_qubit, pbc=self.hamiltonian_['pbc'])
        self.h = tc.quantum.heisenberg_hamiltonian(self.lattice, hzz=self.hamiltonian_['hzz'],
                                                   hxx=self.hamiltonian_['hxx'], hyy=self.hamiltonian_['hyy'],
                                                   hx=self.hamiltonian_['hx'], hy=self.hamiltonian_['hy'],
                                                   hz=self.hamiltonian_['hz'], sparse=self.hamiltonian_['sparse'])
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
            self.two_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.two_qubit_channel_depolarizing_p/15, 2)
            tc.channels.kraus_identity_check(self.two_qubit_dep_channel)
            self.single_qubit_dep_channel = tc.channels.generaldepolarizingchannel(self.single_qubit_channel_depolarizing_p/3, 1)
            tc.channels.kraus_identity_check(self.single_qubit_dep_channel)

    def compute_energy(self, param, structure):
        """
        :param param: Circuit Parameters
        :param structure: Circuit
        :return:
        """
        if self.noise:
            K0 = np.array([[1, 0], [0, 1]]) * np.sqrt(1 - self.bit_flip_p)
            K1 = np.array([[0, 1], [1, 0]]) * np.sqrt(self.bit_flip_p)

            c = tc.DMCircuit(self.n_qubit)
            param_index = 0
            for i, gate in enumerate(structure):
                if gate.name == "CNOT":
                    c.cnot(gate.act_on[0], gate.act_on[1])
                    c.general_kraus(self.two_qubit_dep_channel, gate.act_on[0], gate.act_on[1])
                elif gate.name == "Ry":
                    c.ry(gate.act_on[0], theta=param[param_index][0])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                    param_index += 1
                elif gate.name == "Rz":
                    c.rz(gate.act_on[0], theta=param[param_index][0])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                    param_index += 1
                elif gate.name == "Rx":
                    c.rx(gate.act_on[0], theta=param[param_index][0])
                    c.general_kraus(self.single_qubit_dep_channel, gate.act_on[0])
                    param_index += 1
                else:
                    print("invalid gate!")
                    exit(0)
            for q in range(self.n_qubit):
                c.general_kraus([K0, K1], q)

            """Calculate energy"""
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
        for i in range(len(cir)):
            if cir[i].para_gate:
                param_num += 1
        return param_num

    def train_circuit(self, circuit_and_seed):
        single_circuit = circuit_and_seed[0]
        seed = circuit_and_seed[1]
        np.random.seed(seed)
        tf.random.set_seed(seed)

        param_num = self.get_param_num(single_circuit)
        trainer = tc.backend.jit(tc.backend.value_and_grad(self.compute_energy, argnums=0))
        L = self.depth_count(single_circuit, self.n_qubit)
        par = np.random.normal(loc=0, scale=1/(8*(L+2)), size=param_num)
        par = par.reshape((param_num, 1))
        param = tf.Variable(
            initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr))
        )
        param_initial = param.numpy()
        e_last = 1000
        energy_epoch = []
        opt = tf.keras.optimizers.Adam(0.05)
        for i in range(self.max_iteration):
            e, grad = trainer(param, single_circuit)
            energy_epoch.append(e.numpy())
            opt.apply_gradients([(grad, param)])
            if i % 100 == 0:
                distance = abs(e_last - e.numpy())
                if distance < 0.0001:
                    # print(distance.max())
                    break
                else:
                    e_last = e.numpy()
        return e.numpy(), param.numpy(), energy_epoch

    def process(self, arcs):
        work_queue = []
        for i in range(0, len(arcs)):
            work_queue.extend([[arcs[i], j] for j in range(0, self.n_runs)])
        # result = self.train_circuit(work_queue[0])
        pool = Pool(processes=self.n_cir_parallel)
        result = pool.map(self.train_circuit, work_queue)
        pool.close()
        pool.join()

        energy, param, energy_epoch = [], [], []
        for part in result:
            energy.append(part[0])
            param.append(part[1])
            energy_epoch.append(part[2])

        energy_f, param_f = [], []
        for i in range(0, len(arcs)):
            index0 = i * self.n_runs
            index1 = index0 + self.n_runs
            best_index = np.argmin(energy[index0:index1])
            best_index = best_index + index0
            energy_f.append(energy[best_index])
            param_f.append(param[best_index])

        return energy_f, param_f, energy_epoch