import os
import tensorcircuit as tc
import numpy as np
import tensorflow as tf
from multiprocessing import Pool
import gc

tc.set_dtype("complex128")

def batch_acc(preds, labels):
    # Convert predictions and labels from one-hot to category index
    preds_class = np.argmax(preds, axis=1)
    labels_class = np.argmax(labels, axis=1)
    # Count the number of correct predictions
    correct_predictions = np.sum(preds_class == labels_class)
    accuracy = correct_predictions / len(labels)
    return accuracy

class ClassifyTrainerNew:
    def __init__(self, arg, task_configs, data_class, noise_param=None):
        self.gate_pool = task_configs['gate_pool']
        self.n_qubit = task_configs['qubit']
        self.n_depth = task_configs['layer']
        self.num_embed_gates = task_configs['num_embed_gates']
        self.parallel = arg.parallel
        self.num_epochs = arg.num_epochs
        self.num_meas_qubits = task_configs['num_meas_qubits']
        self.seed = 0
        self.give_up_rest = False
        self.solution = None
        self.data_class = data_class
        self.batch_size = arg.batch_size
        self.train_lr = arg.train_lr
        self.min_delta = arg.min_delta
        self.patience = arg.patience

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

    def compute_energy(self, x, y, param, structure, embed_flags):
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

    def get_qml_vvag(self):
        qml_vvag_train = tc.backend.vectorized_value_and_grad(self.compute_energy, argnums=(2,),vectorized_argnums=(0, 1), has_aux=True)
        qml_vvag_test = tc.backend.vmap(self.compute_energy, vectorized_argnums=(0, 1))
        qml_vvag_train = tc.backend.jit(qml_vvag_train, static_argnums=(3, 4))
        qml_vvag_test = tc.backend.jit(qml_vvag_test, static_argnums=(3, 4))
        return qml_vvag_train, qml_vvag_test

    def train_circuit(self, work_que):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        tc.set_backend("tensorflow")
        embed_flags = []
        param_flags = []
        total_loss = []
        total_acc = []
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
        L = self.depth_count(single_circuit, self.n_qubit)
        par = np.random.normal(loc=0, scale=1/(4*self.n_qubit*(L+2)), size=weights_bounds[-1])
        param = tf.Variable(initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr)))

        qml_vvag_train, qml_vvag_test = self.get_qml_vvag()
        opt = tf.keras.optimizers.Adam(self.train_lr)

        # Early stopping parameters
        patience = self.patience
        min_delta = self.min_delta
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            train_loss = 0
            train_ypreds = []
            for step, (xs, ys) in enumerate(train_data_loader):
                (loss, ypreds), grad = qml_vvag_train(xs, ys, param, single_circuit, embed_flags)
                grad = tf.squeeze(grad)
                opt.apply_gradients([(grad, param)])
                ypreds_numpy = [ypreds[i].numpy() for i in range(len(ypreds))]
                train_ypreds.extend(ypreds_numpy)
                train_loss += tc.backend.sum(loss).numpy()
            total_loss.append(train_loss / len(x_train))
            acc_temp = batch_acc(train_ypreds, y_train)
            total_acc.append(acc_temp)

            # Call garbage collector to free memory
            del xs, ys, loss, ypreds, grad
            gc.collect()

            # Early stopping logic
            if abs(total_loss[epoch] - best_loss) < min_delta:
                patience_counter += 1
            else:
                best_loss = total_loss[epoch]
                patience_counter = 0

            if patience_counter >= patience:
                break
        loss_test, ypreds_test = qml_vvag_test(x_test, y_test, param, single_circuit, embed_flags)
        ypreds_test_numpy = [ypreds_test[i].numpy() for i in range(len(ypreds_test))]
        acc = batch_acc(ypreds_test_numpy, y_test)
        return total_loss, total_acc, acc, ypreds_test_numpy

    def process(self, arcs, inputs_bounds, weights_bounds):
        work_queue = []
        total_loss, total_acc, acc, ypre_test = [], [], [], []
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
        return total_loss, total_acc, acc, ypre_test



