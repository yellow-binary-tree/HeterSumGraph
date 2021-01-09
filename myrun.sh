#!/usr/bin/env bash

# run this script like:
# nohup bash myrun.sh run HDSG wiki_winsize5 5 10 true "use-bert-baseline" 6 > wiki5_bert.log 2>&1 &
# nohup bash myrun.sh run HDSG wiki_winsize5 5 10 false "no-bert-baseline-hidden256" 6 > wiki5_hidden256.log 2>&1 &

export LD_LIBRARY_PATH=/opt/cuda-10.0/lib64:$LD_LIBRARY_PATH;

mode=$1
model=$2
dataset=$3
winsize=$4
winsize_chap_sents=$5
use_bert_embedding=$6
message=$7

test_save_path=$8
test_model=$9

gpu="${!#}"

if [ $winsize == 1 ]; then
    doc_max_timesteps=30
elif [ $winsize == 3 ]; then
    doc_max_timesteps=$(( 30+2*$winsize_chap_sents ))
elif [ $winsize == 5 ]; then
    doc_max_timesteps=$(( 30+4*$winsize_chap_sents ))
elif [ $winsize == 7 ]; then
    doc_max_timesteps=$(( 30+6*$winsize_chap_sents ))
fi

batch_size=16

embedding_path="/share/wangyq/resources/Tencent_AILab_ChineseEmbedding_200w.txt"
vocab_size=100000
eval_iter=$(( 151472/3/$batch_size ))  # eval 3 times per epoch
mmm=5
if [[ $dataset == *"wiki_"* ]]; then
    embedding_path="/share/wangyq/resources/glove.6B.200d.txt"
    batch_size=32
    eval_iter=$(( 38896/2/$batch_size ))  # eval 2 times per epoch
    mmm=1
fi

word_emb_dim=200
if [[ $use_bert_embedding == 'true' ]]; then
    word_emb_dim=768
    embedding_path="/share/wangyq/project/HeterSumGraph/cache/$dataset/embedding"
fi

hidden_size=256

time=$(date "+%Y%m%d_%H%M%S")

if [ $mode == 'debug' ]; then
    echo 'run.sh: train in debug mode '$model $dataset $winsize $gpu
    CUDA_LAUNCH_BLOCKING=1 python -u train.py \
        --model $model --use_bert_embedding $use_bert_embedding \
        --data_dir /share/wangyq/project/HeterSumGraph/data/$dataset \
        --cache_dir /share/wangyq/project/HeterSumGraph/cache/$dataset \
        --save_root save/$time --log_root log \
        --n_feature_size 8 --hidden_size 8 --ffn_inner_hidden_size 8 --lstm_hidden_state 8 \
        --embedding_path ${embedding_path}_debug --word_emb_dim $word_emb_dim \
        --vocab_size $vocab_size --batch_size 4 \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps \
        --lr_descent --grad_clip -m $mmm --eval_after_iterations 100 \
        --cuda --gpu $gpu
elif [ $mode == 'run' ]; then
    echo 'run.sh: train '$model $dataset $winsize $gpu
        # --model $model --use_bert_embedding $use_bert_embedding \
    python -u train.py --model $model \
        --exp_name myHeterSumGraph_${model}_${dataset}_use-bert-embedding_${use_bert_embedding}_${message} \
        --data_dir /share/wangyq/project/HeterSumGraph/data/$dataset \
        --cache_dir /share/wangyq/project/HeterSumGraph/cache/$dataset \
        --log_root log --save_root save/$time \
        --embedding_path $embedding_path --word_emb_dim $word_emb_dim \
        --vocab_size $vocab_size --batch_size $batch_size \
        --n_feature_size $hidden_size --lstm_hidden_state $hidden_size \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps \
        --lr_descent --grad_clip -m $mmm --eval_after_iterations $eval_iter \
        --cuda --gpu $gpu
        # --save_root save/20210109_174225 --restore_model iter_4860 \
        # --save_root save/20201225_230027 --restore_model iter_56790 --start_iteration 56790 \

elif [ $mode == 'test' ]; then
    echo 'run.sh: test '$model $dataset $winsize $gpu
    python -u evaluation.py \
        --model $model --use_bert_embedding $use_bert_embedding \
        --exp_name myHeterSumGraph_test_${model}_${dataset}_use-bert-embedding_${use_bert_embedding}\
        --data_dir /share/wangyq/project/HeterSumGraph/data/$dataset \
        --cache_dir /share/wangyq/project/HeterSumGraph/cache/$dataset \
        --save_root save/$test_save_path --log_root log/ --test_model $test_model \
        --embedding_path $embedding_path --word_emb_dim $word_emb_dim \
        --vocab_size $vocab_size --batch_size $batch_size \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps -m $mmm \
        --save_label --cuda --gpu $gpu
else
    echo 'please select a run mode: debug / run'
fi

exit
