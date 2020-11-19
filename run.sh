#!/usr/bin/env bash

# run this script like:
# nohup bash run.sh debug HSG winsize1 1 1 > HDSG_1118_debug.log 2>&1 &

mode=$1
model=$2
dataset=$3
winsize=$4
gpu=$5

if [ $winsize == 1 ]; then
    doc_max_timesteps=70
elif [ $winsize == 3 ]; then
    doc_max_timesteps=150
elif [ $winsize == 5]; then
    doc_max_timesteps=230
elif [ $winsize == 7]; then
    doc_max_timesteps=310
fi

batch_size=32
eval_iter=6400

time=$(date "+%Y%m%d_%H%M%S")

if [ $mode == 'debug' ]; then
    echo 'run.sh: train in debug mode '$model $dataset $winsize $gpu
    CUDA_LAUNCH_BLOCKING=1 python -u train.py \
        --model $model \
        --data_dir data/$dataset --cache_dir cache/$dataset \
        --save_root save/$time --log_root log \
        --n_feature_size 8 --hidden_size 8 --ffn_inner_hidden_size 8 --lstm_hidden_state 8 \
        --embedding_path Tencent_AILab_ChineseEmbedding_debug.txt --word_emb_dim 200 \
        --vocab_size 100000 --batch_size 1 \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps \
        --lr_descent --grad_clip -m 5 --eval_after_iterations 50 \
        --cuda --gpu $gpu
elif [ $mode == 'run' ]; then
    echo 'run.sh: train '$model $dataset $winsize $gpu
    python -u train.py \
        --model $model \
        --data_dir data/$dataset --cache_dir cache/$dataset \
        --save_root save/$time --log_root log \
        --embedding_path Tencent_AILab_ChineseEmbedding.txt --word_emb_dim 200 \
        --vocab_size 100000 --batch_size $batch_size \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps \
        --lr_descent --grad_clip -m 5 --eval_after_iterations $eval_iter \
        --cuda --gpu $gpu
else
    echo 'please select a run mode: debug / run'
fi

exit
