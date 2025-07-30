
# pre-training for QNN
python MoE_QAS.py --pretraining True --train_N_pretraining 10000 --train_bs_for_pretraining 128 --seed 32 --search_space 'block'
