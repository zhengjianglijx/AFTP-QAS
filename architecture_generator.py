"""
Randomly generate candidate circuits from search space
"""
import numpy as np
import networkx as nx
import utils
from tqdm import tqdm
import pennylane as qml
import matplotlib.pyplot as plt
from quantum_gates import Gate
import argparse
import os
from task_configs import configs


class ArchitectureGenerator:
    """
    Blockwise the search space for sampling
    """
    def __init__(self, gate_pool, two_gate_num_range, num_layers, num_qubits,
                 not_first, start_with_u, num_embed_gates, quantum_device_qbit=None, quantum_device_connection=None):
        self.gate_pool = gate_pool
        self.two_gate_num_range = two_gate_num_range
        self.N = num_qubits
        self.num_embed_gates = num_embed_gates
        if num_layers < 0:
            self.D = max(self.two_gate_num_range) * 5 + self.N * 2
        else:
            self.D = num_layers  #
        self.start_with_u = start_with_u
        self.start_gate = None
        self.not_first = not_first
        self.quantum_device_qbit = quantum_device_qbit
        self.quantum_device_connection = quantum_device_connection

    #  Draw the circuit diagram
    def draw_plot(self, arcs, index, file_name, search_space, seed):
        dev = qml.device('default.qubit', wires=self.N)

        @qml.qnode(dev)
        def circuit(cir):
            for gate in cir:
                if gate.name == 'Hadamard':
                    qml.Hadamard(wires=gate.act_on[0])
                elif gate.name == 'Rx':
                    qml.RX(0, wires=gate.act_on[0])
                elif gate.name == 'Ry':
                    qml.RY(0, wires=gate.act_on[0])
                elif gate.name == 'Rz':
                    qml.RZ(0, wires=gate.act_on[0])
                elif gate.name == 'XX':
                    qml.IsingXX(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'YY':
                    qml.IsingYY(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'ZZ':
                    qml.IsingZZ(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'SWAP':
                    qml.SWAP(wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'U3':
                    qml.U3(0, 0, 0, wires=gate.act_on[0])
                elif gate.name == 'CZ':
                    qml.CZ(wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'CNOT':
                    qml.CNOT(wires=[gate.act_on[0], gate.act_on[1]])
                else:
                    print('invalid gate')
                    exit(0)
            return [qml.expval(qml.PauliZ(q)) for q in range(0, self.N)]

        fig, ax = qml.draw_mpl(circuit)(arcs)
        s_p = f'generated_cir/{file_name}/qubit-{self.N}/img_cir/list_arc_{search_space}_seed{seed}'
        if not os.path.exists(s_p):
            os.makedirs(s_p)
        plt.savefig(f'{s_p}/{index}.png')
        plt.close()
        return 0

    def is_connected(self, graph):
        """Check whether the graph is connected"""
        return nx.is_connected(graph)

    def generate_connected_subgraph(self, graph, num_nodes):
        """
        From the given graph try to generate a connected subgraph with a specified number of nodes.
        graph (NetworkX graph): The given graph.
        num_nodes (int): The number of nodes in the desired connected subgraph.
        Returns:
            subgraph (NetworkX graph): This is a connected subgraph with the specified number of nodes.
            subgraph_nodes (list): This is a list of nodes in the subgraph.
        """
        if num_nodes > graph.number_of_nodes():
            return graph, list(graph.nodes())

        while True:
            nodes = list(graph.nodes())
            selected_nodes = np.random.choice(nodes, num_nodes, replace=False)
            subgraph = graph.subgraph(selected_nodes)
            if self.is_connected(subgraph):
                return subgraph, selected_nodes.tolist()

    def generate_qubit_mappings(self):
        """
        quantum_device_connection: This is a list of connection relationships for the quantum device
        qubit_num: The number of logical bits
        return: mapping: {physical_qubits: logical_qubits}
        """
        device_graph = nx.Graph(self.quantum_device_connection)
        subgraph, subgraph_nodes = self.generate_connected_subgraph(device_graph, self.N)
        physical_qubits = subgraph_nodes
        mapping = {physical_qubits[i]: i for i in range(self.N)}
        logical_connections = [np.sort([mapping[edge[0]], mapping[edge[1]]]) for edge in subgraph.edges()]
        return mapping, logical_connections

    def generate_circuit(self, generation_type, edges_mapping):
        ciru = []
        two_gate_num = np.random.randint(self.two_gate_num_range[0], self.two_gate_num_range[1] + 1)

        if self.start_with_u:
            for i in range(0,len(self.start_gate)):
                if self.start_gate[i]["qubits"] == 1:
                    for j in range(0, self.N):
                        h_g = Gate(**self.start_gate[i])
                        h_g.act_on = [j]
                        ciru.append(h_g)
                else:
                    print('Start gate supposed to be single qubit gate or two-qubit gate!')
                    exit(0)

        if generation_type == 0:
            for i in range(0, two_gate_num):
                gs = self.add_gate(edges_mapping)
                ciru.extend(gs)
        else:
            print('invalid generation type,supposed to be gate_wise only')
            exit(0)

        return ciru

    def add_gate(self, edges_mapping):
        res = []
        positions = np.random.choice(a=len(edges_mapping), size=1, replace=False).item()
        position_1, position_2 = (int(edges_mapping[positions][0]), int(edges_mapping[positions][1]))
        res_1 = Gate(**self.gate_pool[0])
        res_2_1 = Gate(**self.gate_pool[1])
        res_2_2 = Gate(**self.gate_pool[1])
        res_3_1 = Gate(**self.gate_pool[2])
        res_3_2 = Gate(**self.gate_pool[3])
        res_1.act_on = edges_mapping[positions].tolist()
        res_2_1.act_on = [position_1]
        res_2_2.act_on = [position_2]
        res_3_1.act_on = [position_1]
        res_3_2.act_on = [position_2]
        res.append(res_1)
        res.append(res_2_1)
        res.append(res_2_2)
        res.append(res_3_1)
        res.append(res_3_2)
        return res

    def check_same(self, cir1, cir2):
        """
        Check whether two quantum circuits are structurally the same.
        This comparison checks:
            The number of gates
            The name/type of each gate
            The qubits each gate acts on (their positions)
        Parameters:
            cir1: First quantum circuit (list of gate objects).
            cir2: Second quantum circuit (list of gate objects).
        Returns:
            same (bool): True if the circuits are identical in gate sequence and structure; False otherwise.
        """
        if len(cir1) != len(cir2):
            return False
        same = True
        for i in range(0, len(cir1)):
            if cir1[i].name != cir2[i].name:
                same = False
                break
            if cir1[i].act_on != cir2[i].act_on:
                same = False
                break
        return same

    def check(self, cir):
        """
        Check whether the given quantum circuit satisfies the predefined constraints.
        Parameters:
            cir: A list of quantum gates representing the quantum circuit.
        Returns:
            keep (bool): True if the circuit meets the constraints (e.g., max depth), False otherwise.
        """
        res = [0] * self.N
        no_para = 0
        num_two_qubit_gates = 0
        keep = True

        for gate in cir:
            if not gate.para_gate:
                no_para += 1

            if gate.qubits > 1:
                num_two_qubit_gates += 1
                depth_q = []
                for q in gate.act_on:
                    depth_q.append(res[q])
                max_depth = max(depth_q)
                max_depth += 1
                for q in gate.act_on:
                    res[q] = max_depth
            else:
                res[gate.act_on[0]] += 1

        for i in res:
            if i > self.D:
                keep = False
                break
        return keep

    def get_architectures(self, num_architecture, generate_type):
        cirs = []
        num = 0
        pbar = tqdm(total=num_architecture, desc='Randomly generating circuits')
        edge_mapping_list = []
        qubit_mapping_list = []
        inputs_bounds_list = []
        weights_bounds_list = []
        while num < num_architecture:
            inputs_bounds = [0]
            weights_bounds = [0]
            param_indices = []

            qubit_mapping, edges_mapping = self.generate_qubit_mappings()
            temp = self.generate_circuit(generate_type, edges_mapping)

            keep = self.check(temp)
            if keep:
                temp = self.make_it_unique(temp)
                check_same = False
                for c in cirs:
                    check_same = self.check_same(c, temp)
                    if check_same:
                        break
                if not check_same:
                    cirs.append(temp)
                    edge_mapping_list.append(edges_mapping)
                    qubit_mapping_list.append(qubit_mapping)
                    num += 1
                    pbar.update(1)

                    for i in range(len(temp)):
                        if temp[i].para_gate:
                            param_indices.append(i)
                    embeds_indices = np.random.choice(param_indices, self.num_embed_gates, False)

                    for i in range(len(temp)):
                        if i not in param_indices:
                            inputs_bounds.append(inputs_bounds[-1])
                            weights_bounds.append(weights_bounds[-1])
                        else:
                            if i in embeds_indices:
                                inputs_bounds.append(inputs_bounds[-1] + 1)
                                weights_bounds.append(weights_bounds[-1])
                            else:
                                inputs_bounds.append(inputs_bounds[-1])
                                weights_bounds.append(weights_bounds[-1] + 1)

                    inputs_bounds_list.append(inputs_bounds)
                    weights_bounds_list.append(weights_bounds)

        return cirs, edge_mapping_list, qubit_mapping_list, inputs_bounds_list, weights_bounds_list

    def make_it_unique(self, arc):
        """
        Ensures a unique linearized representation of a quantum circuit.
        The method aligns multi-qubit gates in time (or depth) across all their acting qubits,
        so that gates do not overlap in the depth schedule of any qubit.
        arc:
           a circuit in list format.
        Returns:
            list: a list of rearranged gate sequence where all gates are moved to the far left,.
        """
        lists = []
        final_list = []

        for i in range(0, self.N):
            lists.append([])

        for gate in arc:
            if len(gate.act_on) > 1:
                depth_now = []
                for act_q in gate.act_on:
                    depth_now.append(len(lists[act_q]))
                max_depth = max(depth_now)
                for act_q in gate.act_on:
                    while len(lists[act_q]) < max_depth:
                        lists[act_q].append(0)
                min_q = min(gate.act_on)
                lists[min_q].append(gate)
                max_depth_now = len(lists[min_q])
                for act_q in gate.act_on:
                    while len(lists[act_q]) < max_depth_now:
                        lists[act_q].append(0)
            else:
                lists[gate.act_on[0]].append(gate)


        depth = []
        for i in range(0, len(lists)):
            depth.append(len(lists[i]))
        max_depth = max(depth)
        for q in range(self.N):
            while len(lists[q]) < max_depth:
                lists[q].append(0)
        for i in range(max_depth):
            for j in range(0, len(lists)):
                if lists[j][i] != 0:
                    final_list.append(lists[j][i])

        return final_list

    def list_to_adj(self, data):
        res = []
        for i, list_arc in tqdm(enumerate(data), desc='list to adj'):

            temp_op = []
            graph = nx.DiGraph()

            graph.add_node('start', label='start')
            for j in range(0, len(list_arc)):
                graph.add_node(j, label=list_arc[j])
            graph.add_node('end', label='end')

            last = ['start' for _ in range(self.N)]

            for k in range(0, len(list_arc)):
                if list_arc[k].qubits == 1:
                    graph.add_edge(last[list_arc[k].act_on[0]], k)
                    last[list_arc[k].act_on[0]] = k
                else:
                    graph.add_edge(last[list_arc[k].act_on[0]], k)
                    graph.add_edge(last[list_arc[k].act_on[1]], k)
                    last[list_arc[k].act_on[0]] = k
                    last[list_arc[k].act_on[1]] = k

            for _ in last:
                graph.add_edge(_, 'end')

            #  encoding
            for node in graph.nodes:
                if node == 'start':
                    t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                    t2 = [1 for _ in range(self.N)]
                    t1[0] = 1
                    t1.extend(t2)
                    temp_op.append(t1)
                elif node == 'end':
                    t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                    t2 = [1 for _ in range(self.N)]
                    t1[-1] = 1
                    t1.extend(t2)
                    temp_op.append(t1)
                else:
                    if graph.nodes[node]['label'].name == 'CNOT':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[1] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t2[int(graph.nodes[node]['label'].act_on[1])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Ry':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[2] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Rz':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[3] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Rx':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[4] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)

            temp_adj = nx.adjacency_matrix(graph).todense()
            # temp_adj = temp_adj.getA()
            temp_op = np.array(temp_op)
            res.append([temp_adj, temp_op])

        return res


class CircuitGeneratorGatewise:
    """
    Gatewise the search space for sampling
    """
    def __init__(self, gate_pool, gate_num_range, num_layers, num_qubits,
                 no_parameter_gate, max_two_qubit_gates_rate, quantum_device_connection, num_embed_gates):
        self.mean = 0
        self.standard_deviation = 1.35
        self.gate_pool = gate_pool
        self.p1 = 0.3
        self.gate_num_range = gate_num_range
        self.D = num_layers
        self.N = num_qubits
        self.no_parameter_gate = no_parameter_gate
        self.quantum_device_connection = quantum_device_connection
        self.max_two_qubit_gates_rate = max_two_qubit_gates_rate
        self.num_embed_gates = num_embed_gates
        self.seed = 0

    # Draw the circuit diagram
    def draw_plot(self, arcs, index, file_name, search_space, seed):
        dev = qml.device('default.qubit', wires=self.N)

        @qml.qnode(dev)
        def circuit(cir):
            for gate in cir:
                if gate.name == 'Hadamard':
                    qml.Hadamard(wires=gate.act_on[0])
                elif gate.name == 'Rx':
                    qml.RX(0, wires=gate.act_on[0])
                elif gate.name == 'Ry':
                    qml.RY(0, wires=gate.act_on[0])
                elif gate.name == 'Rz':
                    qml.RZ(0, wires=gate.act_on[0])
                elif gate.name == 'XX':
                    qml.IsingXX(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'YY':
                    qml.IsingYY(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'ZZ':
                    qml.IsingZZ(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'SWAP':
                    qml.SWAP(wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'U3':
                    qml.U3(0, 0, 0, wires=gate.act_on[0])
                elif gate.name == 'CZ':
                    qml.CZ(wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'CNOT':
                    qml.CNOT(wires=[gate.act_on[0], gate.act_on[1]])
                else:
                    print('invalid gate')
                    exit(0)
            return [qml.expval(qml.PauliZ(q)) for q in range(0, self.N)]

        fig, ax = qml.draw_mpl(circuit)(arcs)
        # plt.show()
        s_p = f'generated_cir/{file_name}/qubit-{self.N}/img_cir/list_arc_{search_space}_seed{seed}'
        if not os.path.exists(s_p):
            os.makedirs(s_p)
        plt.savefig(f'{s_p}/{index}.png')
        plt.close()
        return 0

    def is_connected(self, graph):
        """ Check if graph is connected """
        return nx.is_connected(graph)

    def generate_connected_subgraph(self, graph, num_nodes):
        """
        From the given graph try to generate a connected subgraph with a specified number of nodes.
        graph (NetworkX graph): The given graph.
        num_nodes (int): The number of nodes in the desired connected subgraph.
        Returns:
            subgraph (NetworkX graph): This is a connected subgraph with the specified number of nodes.
            subgraph_nodes (list): This is a list of nodes in the subgraph.
        """
        if num_nodes > graph.number_of_nodes():
            return graph, list(graph.nodes())

        while True:
            nodes = list(graph.nodes())
            selected_nodes = np.random.choice(nodes, num_nodes, replace=False)
            subgraph = graph.subgraph(selected_nodes)
            if self.is_connected(subgraph):
                return subgraph, selected_nodes.tolist()

    def generate_qubit_mappings(self):
        """
        quantum_device_connection: This is a list of connection relationships for the quantum device
        qubit_num: The number of logical bits
        """
        device_graph = nx.Graph(self.quantum_device_connection)
        subgraph, subgraph_nodes = self.generate_connected_subgraph(device_graph, self.N)
        # Assign a physical qubit to each logical qubit based on the order of the physical qubit list
        physical_qubits = subgraph_nodes
        mapping = {physical_qubits[i]: i for i in range(self.N)}
        logical_connections = [np.sort([mapping[edge[0]], mapping[edge[1]]]) for edge in subgraph.edges()]
        return mapping, logical_connections

    def check_reasonable(self, gate, last_gate_list):
        reasonable = True
        count = 0
        for act_q in gate.act_on:
            if last_gate_list[act_q] == 0:
                break
            elif last_gate_list[act_q].name == gate.name:
                if last_gate_list[act_q].act_on == gate.act_on:
                    count += 1
        if count == len(gate.act_on):
            reasonable = False
        return reasonable

    def generate_circuit(self, generation_type, edges_mapping):
        normal = np.random.normal(self.mean, self.standard_deviation, len(self.gate_pool))
        log_it_list = normal
        log_it_list = np.exp(log_it_list) / sum(np.exp(log_it_list))
        ciru = []
        last_gate_on_each_qubit = [0] * self.N
        gate_num = np.random.randint(self.gate_num_range[0], self.gate_num_range[1] + 1)
        if generation_type == 0:
            cir, last = self.add_gate(0, log_it_list, edges_mapping)
            cir_flag = self.check_reasonable(cir, last_gate_on_each_qubit)
            if cir_flag:
                for q in cir.act_on:
                    last_gate_on_each_qubit[q] = cir
                ciru.append(cir)

            while len(ciru) < gate_num:
                cir, last = self.add_gate(last, log_it_list, edges_mapping)
                cir_flag = self.check_reasonable(cir, last_gate_on_each_qubit)
                if cir_flag:
                    for q in cir.act_on:
                        last_gate_on_each_qubit[q] = cir
                    ciru.append(cir)
        else:
            print('invalid generation type,supposed to be gate_wise only')

        return ciru, gate_num

    def add_gate(self, last_one, log_it_list, edges_mapping):
        if last_one == 0:
            gate = np.random.choice(a=self.gate_pool, size=1, p=log_it_list).item()

        else:
            if np.random.uniform() < self.p1:
                gate = last_one
            else:
                gate = np.random.choice(a=self.gate_pool, size=1, p=log_it_list).item()

        if gate['qubits'] > 1:
            position = np.random.choice(a=len(edges_mapping), size=1).item()
            position = edges_mapping[position]
            res = Gate(**gate)
            res.act_on = [int(position[0]), int(position[1])]
        else:
            position = np.random.choice(a=self.N, size=1).item()
            res = Gate(**gate)
            res.act_on = [position]
        last_one = gate
        return res, last_one

    def check(self, cir):

        res = [0] * self.N
        no_para = 0
        num_two_qubit_gates = 0
        keep = True

        for gate in cir:
            if not gate.para_gate:
                no_para += 1

            if gate.qubits > 1:
                num_two_qubit_gates += 1
                depth_q = []
                for q in gate.act_on:
                    depth_q.append(res[q])
                max_depth = max(depth_q)
                max_depth += 1
                for q in gate.act_on:
                    res[q] = max_depth
            else:
                res[gate.act_on[0]] += 1

        for i in res:
            if i > self.D:
                keep = False
                # print('bad candidate circuit')
                break
        if no_para >= len(cir):
            keep = False
        if num_two_qubit_gates > int(len(cir)*self.max_two_qubit_gates_rate):
            keep = False
        return keep

    def check_same(self, cir1, cir2):
        if len(cir1) != len(cir2):
            return False
        same = True
        for i in range(0, len(cir1)):
            if cir1[i].name != cir2[i].name:
                same = False
                break
            if cir1[i].act_on != cir2[i].act_on:
                same = False
                break

        return same

    def get_architectures(self, num_architecture, generate_type):
        cirs = []
        num = 0
        pbar = tqdm(total=num_architecture, desc='Randomly generating circuits')
        edge_mapping_list = []
        qubit_mapping_list = []
        inputs_bounds_list = []
        weights_bounds_list = []
        while num < num_architecture:

            inputs_bounds = [0]
            weights_bounds = [0]
            param_indices = []

            qubit_mapping, edges_mapping = self.generate_qubit_mappings()
            temp, gate_num = self.generate_circuit(generation_type=generate_type, edges_mapping=edges_mapping)

            keep = self.check(temp)
            if keep:
                temp = self.make_it_unique(temp)
                check_same = False
                for c in cirs:
                    check_same = self.check_same(c, temp)
                    if check_same:
                        break
                if not check_same:
                    cirs.append(temp)
                    edge_mapping_list.append(edges_mapping)
                    qubit_mapping_list.append(qubit_mapping)
                    num += 1
                    pbar.update(1)

                    for i in range(len(temp)):
                        if temp[i].para_gate == 1:
                            param_indices.append(i)
                    embeds_indices = np.random.choice(param_indices, self.num_embed_gates, False)

                    for i in range(len(temp)):
                        if i not in param_indices:
                            inputs_bounds.append(inputs_bounds[-1])
                            weights_bounds.append(weights_bounds[-1])
                        else:
                            if i in embeds_indices:
                                inputs_bounds.append(inputs_bounds[-1] + 1)
                                weights_bounds.append(weights_bounds[-1])
                            else:
                                inputs_bounds.append(inputs_bounds[-1])
                                weights_bounds.append(weights_bounds[-1] + 1)

                    inputs_bounds_list.append(inputs_bounds)
                    weights_bounds_list.append(weights_bounds)

        return cirs, edge_mapping_list, qubit_mapping_list, inputs_bounds_list, weights_bounds_list


    def list_to_adj(self, data):
        res = []
        for i, list_arc in tqdm(enumerate(data), desc='list to adj'):

            temp_op = []
            graph = nx.DiGraph()

            graph.add_node('start', label='start')
            for j in range(0, len(list_arc)):
                graph.add_node(j, label=list_arc[j])
            graph.add_node('end', label='end')

            last = ['start' for _ in range(self.N)]

            for k in range(0, len(list_arc)):
                if list_arc[k].qubits == 1:
                    graph.add_edge(last[list_arc[k].act_on[0]], k)
                    last[list_arc[k].act_on[0]] = k
                else:
                    graph.add_edge(last[list_arc[k].act_on[0]], k)
                    graph.add_edge(last[list_arc[k].act_on[1]], k)
                    last[list_arc[k].act_on[0]] = k
                    last[list_arc[k].act_on[1]] = k

            for _ in last:
                graph.add_edge(_, 'end')

            #  encoding
            for node in graph.nodes:
                if node == 'start':
                    t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                    t2 = [1 for _ in range(self.N)]
                    t1[0] = 1
                    t1.extend(t2)
                    temp_op.append(t1)
                elif node == 'end':
                    t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                    t2 = [1 for _ in range(self.N)]
                    t1[-1] = 1
                    t1.extend(t2)
                    temp_op.append(t1)
                else:
                    if graph.nodes[node]['label'].name == 'CNOT':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[1] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t2[int(graph.nodes[node]['label'].act_on[1])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Ry':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[2] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Rz':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[3] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Rx':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[4] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)

            temp_adj = nx.adjacency_matrix(graph).todense()
            # temp_adj = temp_adj.getA()

            temp_op = np.array(temp_op)

            res.append([temp_adj, temp_op])
        return res

    def make_it_unique(self, arc):
        lists = []
        final_list = []

        for i in range(0, self.N):
            lists.append([])

        for gate in arc:
            if len(gate.act_on) > 1:
                depth_now = []
                for act_q in gate.act_on:
                    depth_now.append(len(lists[act_q]))
                max_depth = max(depth_now)
                for act_q in gate.act_on:
                    while len(lists[act_q]) < max_depth:
                        lists[act_q].append(0)
                min_q = min(gate.act_on)
                lists[min_q].append(gate)
                max_depth_now = len(lists[min_q])
                for act_q in gate.act_on:
                    while len(lists[act_q]) < max_depth_now:
                        lists[act_q].append(0)
            else:
                lists[gate.act_on[0]].append(gate)

        """ Here, all the gates have been added, now we need to arrange the line from left to right and top to bottom. """
        depth = []
        for i in range(0, len(lists)):
            depth.append(len(lists[i]))
        max_depth = max(depth)
        for q in range(self.N):
            while len(lists[q]) < max_depth:
                lists[q].append(0)
        for i in range(max_depth):
            for j in range(0, len(lists)):
                if lists[j][i] != 0:
                    final_list.append(lists[j][i])

        return final_list


class CircuitGeneratorLayerwise:
    """
    Layerwise the search space for sampling
    """
    def __init__(self, gate_pool, gate_num_range, num_layers, num_qubits,
                 no_parameter_gate, max_two_qubit_gates_rate, num_embed_gates):
        self.mean = 0
        self.standard_deviation = 1.35
        self.gate_pool = gate_pool
        self.p1 = 0.7
        self.p2 = 0.65
        self.gate_num_range = gate_num_range
        self.D = num_layers
        self.N = num_qubits
        self.no_parameter_gate = no_parameter_gate
        self.max_two_qubit_gates_rate = max_two_qubit_gates_rate
        self.num_embed_gates = num_embed_gates
        self.seed = 0

    def draw_plot(self, arcs, index, file_name, search_space, seed):
        dev = qml.device('default.qubit', wires=self.N)

        @qml.qnode(dev)
        def circuit(cir):
            for gate in cir:
                if gate.name == 'Hadamard':
                    qml.Hadamard(wires=gate.act_on[0])
                elif gate.name == 'Rx':
                    qml.RX(0, wires=gate.act_on[0])
                elif gate.name == 'Ry':
                    qml.RY(0, wires=gate.act_on[0])
                elif gate.name == 'Rz':
                    qml.RZ(0, wires=gate.act_on[0])
                elif gate.name == 'XX':
                    qml.IsingXX(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'YY':
                    qml.IsingYY(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'ZZ':
                    qml.IsingZZ(0, wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'SWAP':
                    qml.SWAP(wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'U3':
                    qml.U3(0, 0, 0, wires=gate.act_on[0])
                elif gate.name == 'CZ':
                    qml.CZ(wires=[gate.act_on[0], gate.act_on[1]])
                elif gate.name == 'CNOT':
                    qml.CNOT(wires=[gate.act_on[0], gate.act_on[1]])
                else:
                    print('invalid gate')
                    exit(0)
            return [qml.expval(qml.PauliZ(q)) for q in range(0, self.N)]

        fig, ax = qml.draw_mpl(circuit)(arcs)
        # plt.show()
        s_p = f'generated_cir/{file_name}/qubit-{self.N}/img_cir/list_arc_{search_space}_seed{seed}'
        if not os.path.exists(s_p):
            os.makedirs(s_p)
        plt.savefig(f'{s_p}/{index}.png')
        plt.close()
        return 0

    def check_reasonable(self, gates, last_gate_list):
        gate = gates[0]
        reasonable = True
        count = 0
        for act_q in gate.act_on:
            if last_gate_list[act_q] == 0:
                break
            elif last_gate_list[act_q].name == gate.name:
                if last_gate_list[act_q].act_on == gate.act_on:
                    count += 1
        if count == len(gate.act_on):
            reasonable = False
        return reasonable

    def generate_circuit(self, generation_type):
        normal = np.random.normal(self.mean, self.standard_deviation, len(self.gate_pool))
        log_it_list = normal
        log_it_list = np.exp(log_it_list) / sum(np.exp(log_it_list))
        ciru = []
        last_gate_on_each_qubit = [0] * self.N
        gate_num = np.random.randint(self.gate_num_range[0], self.gate_num_range[1] + 1)
        if generation_type == 1:
            while len(ciru) < gate_num:
                cir = self.add_gate(log_it_list)
                cir_flag = self.check_reasonable(cir, last_gate_on_each_qubit)
                if cir_flag:
                    for c in (cir):
                        for q in c.act_on:
                            last_gate_on_each_qubit[q] = c
                    ciru.extend(cir)
        else:
            print('invalid generation type,supposed to be gate_wise only')

        return ciru, gate_num

    def add_gate(self, log_it_list):
        res = []
        gate = np.random.choice(a=self.gate_pool, size=1, p=log_it_list).item()
        position = np.random.choice(a=self.N, size=1).item()
        if gate['qubits'] > 1:
            if position % 2 == 0:
                for i in range(0, int(self.N / 2)):
                    temp = Gate(**gate)
                    temp.act_on = [i * 2, i * 2 + 1]
                    res.append(temp)
            else:
                if self.N % 2 == 0:
                    temp = Gate(**gate)
                    temp.act_on = [self.N - 1, 0]
                    res.append(temp)
                for i in range(1, int(self.N / 2)):
                    temp = Gate(**gate)
                    temp.act_on = [i * 2 - 1, i * 2]
                    res.append(temp)
                if self.N % 2 == 1:
                    temp = Gate(**gate)
                    temp.act_on = [self.N - 2, self.N - 1]
                    res.append(temp)
        else:
            if position % 2 == 0:
                for i in range(0, int(self.N / 2)):
                    temp = Gate(**gate)
                    temp.act_on = [i * 2, i * 2]
                    res.append(temp)
                if self.N % 2 == 1:
                    temp = Gate(**gate)
                    temp.act_on = [self.N - 1, self.N - 1]
                    res.append(temp)
            else:
                for i in range(0, int(self.N / 2)):
                    temp = Gate(**gate)
                    temp.act_on = [i * 2 + 1, i * 2 + 1]
                    res.append(temp)
        return res


    def check(self, cir):

        res = [0] * self.N
        no_para = 0
        num_two_qubit_gates = 0
        keep = True

        for gate in cir:
            if not gate.para_gate:
                no_para += 1

            if gate.qubits > 1:
                num_two_qubit_gates += 1
                depth_q = []
                for q in gate.act_on:
                    depth_q.append(res[q])
                max_depth = max(depth_q)
                max_depth += 1
                for q in gate.act_on:
                    res[q] = max_depth
            else:
                res[gate.act_on[0]] += 1

        for i in res:
            if i > self.D:
                keep = False
                break
        if no_para >= len(cir):
            keep = False
        if num_two_qubit_gates > int(len(cir)*self.max_two_qubit_gates_rate):
            keep = False
        return keep

    def check_same(self, cir1, cir2):
        if len(cir1) != len(cir2):
            return False
        same = True
        for i in range(0, len(cir1)):
            if cir1[i].name != cir2[i].name:
                same = False
                break
            if cir1[i].act_on != cir2[i].act_on:
                same = False
                break

        return same

    def get_architectures(self, num_architecture, generate_type):
        cirs = []
        num = 0
        pbar = tqdm(total=num_architecture, desc='Randomly generating circuits')
        inputs_bounds_list = []
        weights_bounds_list = []
        while num < num_architecture:

            inputs_bounds = [0]
            weights_bounds = [0]
            param_indices = []

            temp, gate_num = self.generate_circuit(generation_type=generate_type)

            keep = self.check(temp)
            if keep:
                temp = self.make_it_unique(temp)
                check_same = False
                for c in cirs:
                    check_same = self.check_same(c, temp)
                    if check_same:
                        break
                if not check_same:
                    cirs.append(temp)
                    num += 1
                    pbar.update(1)

                    for i in range(len(temp)):
                        if temp[i].para_gate == 1:
                            param_indices.append(i)
                    embeds_indices = np.random.choice(param_indices, self.num_embed_gates, False)

                    for i in range(len(temp)):
                        if i not in param_indices:
                            inputs_bounds.append(inputs_bounds[-1])
                            weights_bounds.append(weights_bounds[-1])
                        else:
                            if i in embeds_indices:
                                inputs_bounds.append(inputs_bounds[-1] + 1)
                                weights_bounds.append(weights_bounds[-1])
                            else:
                                inputs_bounds.append(inputs_bounds[-1])
                                weights_bounds.append(weights_bounds[-1] + 1)

                    inputs_bounds_list.append(inputs_bounds)
                    weights_bounds_list.append(weights_bounds)

        return cirs, inputs_bounds_list, weights_bounds_list


    def list_to_adj(self, data):
        res = []
        for i, list_arc in tqdm(enumerate(data), desc='list to adj'):

            temp_op = []
            graph = nx.DiGraph()

            graph.add_node('start', label='start')
            for j in range(0, len(list_arc)):
                graph.add_node(j, label=list_arc[j])
            graph.add_node('end', label='end')

            last = ['start' for _ in range(self.N)]

            for k in range(0, len(list_arc)):
                if list_arc[k].qubits == 1:
                    graph.add_edge(last[list_arc[k].act_on[0]], k)
                    last[list_arc[k].act_on[0]] = k
                else:
                    graph.add_edge(last[list_arc[k].act_on[0]], k)
                    graph.add_edge(last[list_arc[k].act_on[1]], k)
                    last[list_arc[k].act_on[0]] = k
                    last[list_arc[k].act_on[1]] = k

            for _ in last:
                graph.add_edge(_, 'end')

            #  encoding
            for node in graph.nodes:
                if node == 'start':
                    t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                    t2 = [1 for _ in range(self.N)]
                    t1[0] = 1
                    t1.extend(t2)
                    temp_op.append(t1)
                elif node == 'end':
                    t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                    t2 = [1 for _ in range(self.N)]
                    t1[-1] = 1
                    t1.extend(t2)
                    temp_op.append(t1)
                else:
                    if graph.nodes[node]['label'].name == 'CNOT':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[1] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t2[int(graph.nodes[node]['label'].act_on[1])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Ry':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[2] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Rz':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[3] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)
                    if graph.nodes[node]['label'].name == 'Rx':
                        t1 = [0 for _ in range(len(self.gate_pool) + 2)]
                        t2 = [0 for _ in range(self.N)]
                        t1[4] = 1
                        t2[int(graph.nodes[node]['label'].act_on[0])] = 1
                        t1.extend(t2)
                        temp_op.append(t1)

            temp_adj = nx.adjacency_matrix(graph).todense()
            # temp_adj = temp_adj.getA()

            temp_op = np.array(temp_op)

            res.append([temp_adj, temp_op])

        return res

    def make_it_unique(self, arc):
        lists = []
        final_list = []

        for i in range(0, self.N):
            lists.append([])

        for gate in arc:
            if len(gate.act_on) > 1:
                depth_now = []
                for act_q in gate.act_on:
                    depth_now.append(len(lists[act_q]))
                max_depth = max(depth_now)
                for act_q in gate.act_on:
                    while len(lists[act_q]) < max_depth:
                        lists[act_q].append(0)
                min_q = min(gate.act_on)
                lists[min_q].append(gate)
                max_depth_now = len(lists[min_q])
                for act_q in gate.act_on:
                    while len(lists[act_q]) < max_depth_now:
                        lists[act_q].append(0)
            else:
                lists[gate.act_on[0]].append(gate)

        """ Here, all the gates have been added, now we need to arrange the line from left to right and top to bottom. """
        depth = []
        for i in range(0, len(lists)):
            depth.append(len(lists[i]))
        max_depth = max(depth)
        for q in range(self.N):
            while len(lists[q]) < max_depth:
                lists[q].append(0)
        for i in range(max_depth):
            for j in range(0, len(lists)):
                if lists[j][i] != 0:
                    final_list.append(lists[j][i])
        return final_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qubit', type=int, default=4, help='qubit')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--task', type=str, default='Classification', help='see task_congigs.py for available tasks')
    parser.add_argument('--search_space', type=str, default='block', help='')
    parser.add_argument('--task_result', type=str, default='fmnist_4', help='')
    parser.add_argument('--pretraining', type=int, default=0, help='')
    args = parser.parse_args()
    cfgs = configs[args.task_result]

    if args.pretraining == 0:
        s_p_l = f'datasets/{args.task}/{args.search_space}_seed{args.seed}/list_cir_{args.search_space}_seed{args.seed}'
        Number_samples = 11000
    elif args.pretraining == 1:
        args.seed = 32
        Number_samples = 10000
        s_p_l = f'datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/list_cir_{args.search_space}_seed{args.seed}_pretrained'
    else:
        args.seed = 32
        Number_samples = 100000
        s_p_l = f'datasets/{args.task}/{args.search_space}_seed{args.seed}_pretrained/list_cir_{args.search_space}_seed{args.seed}_pretrained_vae'
    np.random.seed(args.seed)
    print(f'seed:{args.seed}')

    if args.search_space == 'block':
        lower_bound = cfgs['lower_bound']
        upper_bound = cfgs['upper_bound']
    else:
        lower_bound = cfgs['lower_bound'] * 5 + 2 * args.qubit
        upper_bound = cfgs['upper_bound'] * 5 + 2 * args.qubit

    print('a')

    if args.search_space == 'block':
        ag = ArchitectureGenerator(gate_pool=[{"name": "CNOT", "qubits": 2, "para_gate": False},
                                              {"name": "Ry", "qubits": 1, "para_gate": True},
                                              {"name": "Rz", "qubits": 1, "para_gate": True},
                                              {"name": "Rx", "qubits": 1, "para_gate": True}],
                                   two_gate_num_range=[lower_bound, upper_bound],
                                   num_layers=-1,
                                   num_qubits=args.qubit,
                                   not_first=[{"name": "CNOT", "qubits": 2, "para_gate": False}],
                                   start_with_u=True,
                                   num_embed_gates=16,
                                   quantum_device_qbit=7,
                                   quantum_device_connection=[[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]]
                                   )
        ag.start_gate = [{"name": "Ry", "qubits": 1, "para_gate": True},
                         {"name": "Rz", "qubits": 1, "para_gate": True}]
        list_arc, edge_mapping_list, qubit_mapping_list, inputs_bounds_list, weights_bounds_list = ag.get_architectures(Number_samples, 0)

    elif args.search_space == 'gatewise':
        ag = CircuitGeneratorGatewise(gate_pool=[{"name": "Ry", "qubits": 1, "para_gate": True},
                                                       {"name": "Rz", "qubits": 1, "para_gate": True},
                                                       {"name": "Rx", "qubits": 1, "para_gate": True},
                                                       {"name": "CNOT", "qubits": 2, "para_gate": False}],
                                            gate_num_range=[lower_bound, upper_bound],
                                            num_layers=upper_bound,
                                            num_qubits=args.qubit,
                                            no_parameter_gate=[],
                                            max_two_qubit_gates_rate=0.5,
                                            quantum_device_connection=[[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]],
                                            num_embed_gates=16)
        ag.start_gate = [{"name": "Ry", "qubits": 1, "para_gate": True},
                         {"name": "Rz", "qubits": 1, "para_gate": True}]
        list_arc, edge_mapping_list, qubit_mapping_list, inputs_bounds_list, weights_bounds_list = ag.get_architectures(Number_samples, 0)
    else:
        ag = CircuitGeneratorLayerwise(gate_pool=[{"name": "Ry", "qubits": 1, "para_gate": True},
                                                        {"name": "Rz", "qubits": 1, "para_gate": True},
                                                        {"name": "Rx", "qubits": 1, "para_gate": True},
                                                        {"name": "CNOT", "qubits": 2, "para_gate": False}],
                                             gate_num_range=[lower_bound, upper_bound],
                                             num_layers=upper_bound,
                                             num_qubits=args.qubit,
                                             no_parameter_gate=[],
                                             max_two_qubit_gates_rate=0.5,
                                             num_embed_gates=16)
        ag.start_gate = [{"name": "Ry", "qubits": 1, "para_gate": True},
                         {"name": "Rz", "qubits": 1, "para_gate": True}]
        list_arc, inputs_bounds_list, weights_bounds_list = ag.get_architectures(Number_samples, 1)
    if not os.path.exists(s_p_l):
        os.makedirs(s_p_l)

    if args.pretraining == 0:
        utils.save_pkl(list_arc, f'{s_p_l}/list_arc_{args.search_space}_seed{args.seed}.pkl')
        utils.save_pkl(inputs_bounds_list, f'{s_p_l}/inputs_bounds_{args.search_space}_seed{args.seed}.pkl')
        utils.save_pkl(weights_bounds_list, f'{s_p_l}/weights_bounds_{args.search_space}_seed{args.seed}.pkl')
    elif args.pretraining == 1:
        utils.save_pkl(list_arc, f'{s_p_l}/list_arc_{args.search_space}_seed{args.seed}_pretrained.pkl')
        utils.save_pkl(inputs_bounds_list, f'{s_p_l}/inputs_bounds_{args.search_space}_seed{args.seed}_pretrained.pkl')
        utils.save_pkl(weights_bounds_list, f'{s_p_l}/weights_bounds_{args.search_space}_seed{args.seed}_pretrained.pkl')
    else:
        utils.save_pkl(list_arc, f'{s_p_l}/list_arc_{args.search_space}_seed{args.seed}_pretrained_vae.pkl')
        utils.save_pkl(inputs_bounds_list, f'{s_p_l}/inputs_bounds_{args.search_space}_seed{args.seed}_pretrained_vae.pkl')
        utils.save_pkl(weights_bounds_list,f'{s_p_l}/weights_bounds_{args.search_space}_seed{args.seed}_pretrained_vae.pkl')
    print('a')
