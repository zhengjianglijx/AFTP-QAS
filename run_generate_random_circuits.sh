# QNN
for (( seed = 0; seed < 5; seed+=1)); do
    python architecture_generator.py --seed $seed --task 'Classification' --qubit 4 --task_result 'fmnist_4' --search_space 'block' --pretraining 0
done

# QNN pre-training
python architecture_generator.py --seed $seed --task 'Classification' --qubit 4 --task_result 'fmnist_4' --search_space 'block' --pretraining 1

# QNN pre-training-vae
python architecture_generator.py --seed $seed --task 'Classification' --qubit 4 --task_result 'fmnist_4' --search_space 'block' --pretraining 2

