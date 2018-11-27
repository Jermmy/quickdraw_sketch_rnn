batch_size=128
input_size=3
hidden_size=256
n_layers=2
load_model=None
lr=1e-4
start_idx=0
epochs=10
rnn_type=gru

ckpt_path=ckpt/${rnn_type}
result_path=result/${rnn_type}

python train.py --batch_size ${batch_size} --input_size ${input_size} --rnn_type ${rnn_type} \
                --hidden_size ${hidden_size} --n_layers ${n_layers} \
                --lr ${lr} --start_idx ${start_idx} --epochs ${epochs} \
                --ckpt_path ${ckpt_path} --result_path ${result_path}