
# MoE_QAS for QNN
for (( seed = 0; seed < 5; seed+=1)); do
    python MoE_QAS.py --seed $seed --search_space 'block' --task_result 'fmnist_4' --method 'NonLiner_QAS' --task 'Classification' --qubit 4
done

python eval_circuits.py --search_space 'block' --task_result 'fmnist_4' --method 'NonLiner_QAS' --task 'Classification'