batch_size=64
input_size=3
hidden_size=256
n_layers=2
lr=1e-4
start_idx=0
epochs=10
rnn_type=gru
avg_out=1
bi_rnn=1
use_conv=0

ckpt_path=ckpt/${rnn_type}/layers_${n_layers}_hidden_${hidden_size}
result_path=result/${rnn_type}/layers_${n_layers}_hidden_${hidden_size}

if [ ${avg_out} == '1' ]; then
    ckpt_path=${ckpt_path}_avgout
    result_path=${result_path}_avgout
fi

if [ ${bi_rnn} == '1' ]; then
    ckpt_path=${ckpt_path}_birnn
    result_path=${result_path}_birnn
fi

if [ ${use_conv} == '1' ]; then
    ckpt_path=${ckpt_path}_useconv
    result_path=${result_path}_useconv
fi

load_model=${ckpt_path}/epoch-${start_idx}.pkl

python3 train.py --batch_size ${batch_size} --input_size ${input_size} \
                --rnn_type ${rnn_type} --avg_out ${avg_out} --bi_rnn ${bi_rnn} --use_conv ${use_conv} \
                --hidden_size ${hidden_size} --n_layers ${n_layers} \
                --lr ${lr} --start_idx ${start_idx} --epochs ${epochs} \
                --ckpt_path ${ckpt_path} --result_path ${result_path} \
                # --load_model ${load_model}