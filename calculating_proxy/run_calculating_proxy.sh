#  "Switch to the directory where the script is located (i.e., calculating_proxy/)"
cd "$(dirname "$0")"

# "Set PYTHONPATH to the parent directory (i.e., the project root)"
export PYTHONPATH=..

# Before running run_calculating_proxy.sh, make sure to first select and prepare the required circuit architecture.
for (( seed = 0; seed < 5; seed+=1)); do
    python Width.py --seed $seed --search_space 'block' --qubits 4 --pretraining 0 --task 'Classification'
    python Depth.py --seed $seed --search_space 'block' --qubits 4 --pretraining 0 --task 'Classification'
    python Expressibility.py --seed $seed --search_space 'block' --qubits 4 --pretraining 0 --task 'Classification'
    python Trainability_task_w.py --seed $seed --search_space 'block' --qubits 4 --pretraining 0 --task 'Classification'
done

# proxy for QNN pre-training
python Width.py --search_space 'block' --qubits 4 --pretraining 1 --task 'Classification' --seed 32
python Depth.py --search_space 'block' --qubits 4 --pretraining 1 --task 'Classification' --seed 32
python Expressibility.py --search_space 'block' --qubits 4 --pretraining 1 --task 'Classification' --seed 32
python Trainability_task_w.py --search_space 'block' --qubits 4 --pretraining 1 --task 'Classification' --seed 32
