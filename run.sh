#!/usr/bin/env bash

# run this script like:
# nohup bash run.sh debug HDSG qidian_1109_seq 7 > HDSG_1114_debug.log 2>&1 &

mode=$1
model=$2
dataset=$3
gpu=$4

task='single'
share_datadir='/share/wangyq/data/qidian_summ/'$dataset
datadir='./data/'$dataset

# if [ ! -d $datadir ]; then
#     mkdir $datadir
# fi

# echo 'run.sh: copying training data to project folder '$dataset
# python MoveTrainingData.py --src_folder $share_datadir --dest_folder $datadir

# echo 'run.sh: prepare dataset '$dataset
# bash PrepareDataset.sh $dataset $datadir $task

# !! check if commented before running !!

if [ $model == 'HSG' ]; then
    max_timesteps=150
    batch_size=32
    eval_iter=5000
elif [ $model == 'HDSG' ]; then
    max_timesteps=550
    batch_size=32
    eval_iter=5000
fi

time=$(date "+%Y%m%d_%H%M%S")

if [ $mode == 'debug' ]; then
    echo 'run.sh: train in debug mode '$model $dataset $gpu
    python -u train.py \
        --model $model \
        --data_dir data/$dataset --cache_dir cache/$dataset \
        --save_root save/$time --log_root log \
        --n_feature_size 8 --hidden_size 8 --ffn_inner_hidden_size 8 --lstm_hidden_state 8 \
        --embedding_path Tencent_AILab_ChineseEmbedding_debug.txt --word_emb_dim 200 \
        --vocab_size 30000 --batch_size 4 --num_workers 1 --doc_max_timesteps $max_timesteps\
        --lr_descent --grad_clip -m 5 --eval_after_iterations $eval_iter \
        --cuda --gpu $gpu
elif [ $mode == 'run' ]; then
    echo 'run.sh: train '$model $dataset $gpu
    python -u train.py \
        --model $model \
        --data_dir data/$dataset --cache_dir cache/$dataset \
        --save_root save/$time --log_root log \
        --embedding_path Tencent_AILab_ChineseEmbedding.txt --word_emb_dim 200 \
        --vocab_size 100000 --batch_size $batch_size --doc_max_timesteps $max_timesteps --num_workers 1 \
        --lr_descent --grad_clip -m 5 --eval_after_iterations $eval_iter \
        --cuda --gpu $gpu
else
    echo 'please select a run mode: debug / run'
fi
