# Adaptive Fusion of Training-free Proxies for Quantum Architecture Search

This repository provides the official implementation of the paper:

> **Adaptive Fusion of Training-free Proxies for Quantum Architecture Search**  
> (Submission to Physical Review Applied)

We introduce an adaptive Quantum Architecture Search (QAS) framework that dynamically fuses multiple training-free proxies using a Mixture-of-Experts model. The framework generalizes across various VQA tasks and circuit search spaces, and significantly improves the efficiency and accuracy of circuit evaluation.

📚 Built upon [TensorCircuit](https://tensorcircuit.readthedocs.io/en/latest/index.html).

---

## 🧩 Key Features

- Dynamically learns to combine multiple training-free proxies for circuit evaluation.
-  Incorporates Expressibility, Trainability, SNIP, Topological Width, and Depth.
-  Supports device-specific constraints (e.g., IBMQ Casablanca).
- Blockwise circuit construction reduces redundancy and significantly shrinks the architecture search space.
- Pretrained experts are transferable across VQA tasks and search spaces.

---

## 📦 Installation

**Prerequisites**

- Python 3.8+

- Pip

  **Install dependencies**

  ```
  pip install -r requirements.txt
  ```

Setup should now be complete.

---

## 🚀 Example Usage

### 1. Generate Candidate Circuits

Generate candidate PQCs for different search spaces:

```bash
run_generate_random_circuits.sh
```

- **Blockwise** circuits (proposed in this work) and **Gatewise** circuits [1] are both generated under hardware constraints, tailored to the topology of the IBMQ Casablanca quantum device.
- In **Layerwise** circuits [1], a specific gate type is selected and applied uniformly across either all odd-indexed or all even-indexed qubits.

---

### 2. Compute Proxy Metrics

Evaluate circuits with various training-free proxy metrics:

```bash
cd calculating_proxy
run_calculating_proxy.sh
cd ..
```

Computed proxies include:

- Expressibility

- Trainability

- SNIP

- Topological Width

- Topological Depth

  These proxies are designed to assess circuit quality from both functional and structural perspectives without requiring task-specific labels or training.

---

### 3. Pretrain

Pretrain expert models (for each proxy) and the gating network:

```bash
run_pre_training.sh
```

---

### 4. Evaluation

Following circuit ranking via the Mixture-of-Experts model, the candidate with the highest predicted performance is selected and undergoes parameter training to assess its actual (ground-truth) performance:

```bash
run_Ours.sh
```

---

### 5. Baseline Methods

Use linear or non-linear static fusion methods [2] to integrate proxies:

```bash
run_Linear.sh
run_Non_Linear.sh
```

## 📁 Project Structure

```
.
├── calculating_proxy        # Implementations of task-agnostic training-free proxies
├── datasets                 # Dataset loaders and metadata for VQA tasks
├── model                    # GIN, and expert model definitions
├── optimize_circuits        # Circuit training for QNN and VQE tasks
├── pretrained               # Pretrained expert models and gating network checkpoints
├── Prune                    # Pruning utilities for quantum circuits
├── requirements.txt         # Python package dependencies
└── README.md                # Project overview and usage instructions
```

---

## 📊 Used VQA Tasks

- Quantum Neural Network (QNN) — MNIST, FMNIST datasets
- Variational Quantum Eigensolver (VQE) — Transverse Field Ising Model (TFIM)

---

## 📜  References

```bibtex
[1] S.-X. Zhang, C.-Y. Hsieh, S. Zhang, and H. Yao, Neural predictor based quantum architecture search, Machine Learning: Science and Technology 2, 045027 (2021).
[2] J. Lee and B. Ham, AZ-NAS: Assembling zero-cost proxies for network architecture search, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024) pp. 5893–5903.
```



