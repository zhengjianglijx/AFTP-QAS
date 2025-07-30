expert_to_remove=(0 1 2 3 4 5)
for k in "${!expert_to_remove[@]}"; do
    for (( seed = 0; seed < 5; seed+=1 )); do
        python MoE_QAS.py --seed $seed --search_space 'block' --task_result 'fmnist_4' --expert_to_remove "${expert_to_remove[k]}" --pre_epochs 50 --method 'MoE_QAS_removal' --num_experts 5
    done
done

python eval_circuits.py --search_space 'block' --task_result 'fmnist_4' --method 'MoE_QAS_removal' --task 'Classification'