#!/usr/bin/env bash

# run this script like:
# nohup bash myrun.sh run HDSG winsize3_bow_cut 3 6 > HDSGfc3_1126.log 2>&1 &

mode=$1
model=$2
dataset=$3
winsize=$4
gpu=$5

test_save_path=$6
test_model=$7

if [ $winsize == 1 ]; then
    doc_max_timesteps=30
elif [ $winsize == 3 ]; then
    doc_max_timesteps=50
elif [ $winsize == 5 ]; then
    doc_max_timesteps=70
elif [ $winsize == 7 ]; then
    doc_max_timesteps=90
fi

batch_size=16
eval_iter=$(( 151000/3/$batch_size ))       # eval 3 times per iteration

time=$(date "+%Y%m%d_%H%M%S")

if [ $mode == 'debug' ]; then
    echo 'run.sh: train in debug mode '$model $dataset $winsize $gpu
    CUDA_LAUNCH_BLOCKING=1 python -u train.py \
        --model $model \
        --data_dir /share/wangyq/project/HeterSumGraph/data/$dataset \
        --cache_dir /share/wangyq/project/HeterSumGraph/cache/$dataset \
        --save_root save/$time --log_root log \
        --n_feature_size 8 --hidden_size 8 --ffn_inner_hidden_size 8 --lstm_hidden_state 8 \
        --embedding_path Tencent_AILab_ChineseEmbedding_debug.txt --word_emb_dim 200 \
        --vocab_size 100000 --batch_size 4 \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps \
        --lr_descent --grad_clip -m 5 --eval_after_iterations 100 \
        --cuda --gpu $gpu
elif [ $mode == 'run' ]; then
    echo 'run.sh: train '$model $dataset $winsize $gpu
    python -u train.py \
        --model $model --exp_name myHeterSumGraph_${model}_${dataset}_${time} \
        --data_dir /share/wangyq/project/HeterSumGraph/data/$dataset \
        --cache_dir /share/wangyq/project/HeterSumGraph/cache/$dataset \
        --save_root save/$time --log_root log \
        --embedding_path Tencent_AILab_ChineseEmbedding_200w.txt --word_emb_dim 200 \
        --vocab_size 100000 --batch_size $batch_size \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps \
        --lr_descent --grad_clip -m 5 --eval_after_iterations $eval_iter \
        --train_num_workers 0 --eval_num_workers 0 \
        --cuda --gpu $gpu
elif [ $mode == 'test' ]; then
    echo 'run.sh: test '$model $dataset $winsize $gpu
    python -u evaluation.py \
        --model $model --exp_name myHeterSumGraph_test_${model}_${dataset}\
        --data_dir /share/wangyq/project/HeterSumGraph/data/$dataset \
        --cache_dir /share/wangyq/project/HeterSumGraph/cache/$dataset \
        --save_root save/$test_save_path --log_root log/ --test_model $test_model \
        --embedding_path Tencent_AILab_ChineseEmbedding_200w.txt --word_emb_dim 200 \
        --vocab_size 100000 --batch_size $batch_size \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps -m 2 \
        --save_label --cuda --gpu $gpu
else
    echo 'please select a run mode: debug / run'
fi

exit
