#!/usr/bin/env bash

GPU_USE='0'

dataset='qidian_sample_seq'  # !! check before running !!
share_datadir='/share/wangyq/data/qidian_summ/'$dataset
datadir='./data/'$dataset

if [ ! -d $datadir ]; then
    mkdir $datadir
fi

echo 'run.sh: copying training data to project folder'
python MoveTrainingData.py --src_folder $share_datadir --dest_folder $datadir

echo 'run.sh: prepare dataset'
bash PrepareDataset.sh $dataset $datadir

# !! check if commented before running !!

if [ $1 == 'debug' ]; then  
    # debug mode
    echo 'run.sh: train in debug mode'
        python train.py \
        --model HSG \
        --data_dir data/$dataset --cache_dir cache/$dataset \
        --save_root save --log_root log \
        --n_feature_size 8 --hidden_size 8 --ffn_inner_hidden_size 8 --lstm_hidden_state 8 \
        --embedding_path Tencent_AILab_ChineseEmbedding.txt --word_emb_dim 200 \
        --vocab_size 100 --batch_size 1 --num_workers 1 \
        --lr_descent --grad_clip -m 5 \
        --cuda --gpu $GPU_USE
elif [ $1 == 'run' ]; then
    # train mode
    echo 'run.sh: train'
    python train.py \
        --model HSG \
        --data_dir data/$dataset --cache_dir cache/$dataset \
        --save_root save --restore_model bestmodel --log_root log \
        --embedding_path Tencent_AILab_ChineseEmbedding.txt --word_emb_dim 200 \
        --vocab_size 100000 --batch_size 32 --num_workers 1 \
        --lr_descent --grad_clip -m 5 \
        --cuda --gpu $GPU_USE
else
    echo 'please select a run mode: debug / run'
fi
