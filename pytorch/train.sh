batch_size=32
input_size=3
hidden_size=100
n_layers=2
load_model=None
lr=1e-3
start_idx=0
epochs=10

ckpt_path=ckpt/gru
result_path=result/gru

python train.py --batch_size ${batch_size} --input_size ${input_size} \
                --hidden_size ${hidden_size} --n_layers ${n_layers} \
                --lr ${lr} --start_idx ${start_idx} --epochs ${epochs} \
                --ckpt_path ${ckpt_path} --result_path ${result_path}