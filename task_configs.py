"""
Configurations for specific tasks
"""
configs = {
    'fmnist_4':
        {'qubit': 4,  # Number of qubits on the line
         'layer': -1,  # Max layers of the line
         'gate_pool': [
             {"name": "CNOT", "qubits": 2, "para_gate": False},
             {"name": "Ry", "qubits": 1, "para_gate": True},
             {"name": "Rz", "qubits": 1, "para_gate": True},
             {"name": "Rx", "qubits": 1, "para_gate": True}],  # Quantum gate pool
         'g_or_l': 0,  # 0,gate_wise;1,layer_wise
         'start_with_u': True,
         'start_gate': [{"name": "Ry", "qubits": 1, "para_gate": True},
                        {"name": "Rz", "qubits": 1, "para_gate": True}],
         'num_meas_qubits': 4,
         'num_embed_gates': 16,
         'num_data_reps': 1,
         'num_classes': 4,
         'file_type': 'txt',
         'available_edge': False,
         'not_first': [{"name": "CNOT", "qubits": 2, "para_gate": False}],  # The set of gates that are not allowed as initial gates
         'quantum_device_qbit': 7,
         'quantum_device_connection': [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]],  # ibmq_casablanca
         'lower_bound': 8,
         'upper_bound': 14
         },
    'mnist_4':
        {'qubit': 4,
         'layer': -1,

         'gate_pool': [
             {"name": "CNOT", "qubits": 2, "para_gate": False},
             {"name": "Ry", "qubits": 1, "para_gate": True},
             {"name": "Rz", "qubits": 1, "para_gate": True},
             {"name": "Rx", "qubits": 1, "para_gate": True}],
         'g_or_l': 0,
         'start_with_u': True,
         'start_gate': [{"name": "Ry", "qubits": 1, "para_gate": True},
                        {"name": "Rz", "qubits": 1, "para_gate": True}],
         # 'available_edge': False,
         'num_meas_qubits': 4,
         'num_embed_gates': 16,
         'num_data_reps': 1,
         'num_classes': 4,
         'file_type': 'txt',
         'not_first': [{"name": "CNOT", "qubits": 2, "para_gate": False}],
         'quantum_device_qbit': 7,
         'quantum_device_connection': [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]],
         'lower_bound': 8,
         'upper_bound': 14
         },
    'TFIM':
        {'qubit': 7,
         'layer': -1,
         'gate_pool': [
             {"name": "CNOT", "qubits": 2, "para_gate": False},
             {"name": "Ry", "qubits": 1, "para_gate": True},
             {"name": "Rz", "qubits": 1, "para_gate": True},
             {"name": "Rx", "qubits": 1, "para_gate": True}],
         'g_or_l': 0,
         'start_with_u': True,
         'start_gate': [{"name": "Ry", "qubits": 1, "para_gate": True},
                        {"name": "Rz", "qubits": 1, "para_gate": True}],
         'Hamiltonian': {'pbc': True, 'hzz': 1, 'hxx': 0, 'hyy': 0, 'hx': 1, 'hy': 0, 'hz': 0, 'sparse': False},
         'theoretical': -8.76257,
         'available_edge': False,
         'not_first': [{"name": "CNOT", "qubits": 2, "para_gate": False}],
         'quantum_device_qbit': 7,
         'quantum_device_connection': [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]],
         'lower_bound': 6,
         'upper_bound': 10
         },
}
