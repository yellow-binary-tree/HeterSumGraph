#!/usr/bin/env bash

dataset="$1"
datadir="$2"
task="$3"


# -u to check bounded args, -e to exit when error
#set -u
set -e

if [ ! -n "$dataset" ]; then
    echo "lose the dataset name!"
    exit
fi

if [ ! -n "$datadir" ]; then
    echo "lose the data directory!"
    exit
fi

if [ ! -n "$task" ]; then
    task=single
fi

type=(train val test)

{
echo -e "\033[34m[Shell] Create Vocabulary! \033[0m"
python -u script/createVoc.py \
    --dataset $dataset \
    --data_path $datadir/train_split
    # --data_path $datadir/train.label.jsonl &
} &
{
echo -e "\033[34m[Shell] Get low tfidf words from training set! \033[0m"
python -u script/lowTFIDFWords.py \
    --dataset $dataset \
    --data_path $datadir/train.label.jsonl
} &

echo -e "\033[34m[Shell] Get word2sent edge feature! \033[0m"
for i in ${type[*]}
    do {
        if [ "$i" == "train" ]; then
        python -u script/calw2sTFIDF.py \
            --dataset $dataset \
            --data_path $datadir/train_split
        else
        python -u script/calw2sTFIDF.py \
            --dataset $dataset \
            --data_path $datadir/$i.label.jsonl
        fi
    } & done

if [ "$task" == "multi" ]; then
    echo -e "\033[34m[Shell] Get word2doc edge feature! \033[0m"
    for i in ${type[*]}
        do {
            if [ "$i" == "train" ]; then
                python -u script/calw2dTFIDF.py \
                    --dataset $dataset \
                    --data_path $datadir/train_split
            else
                python -u script/calw2dTFIDF.py \
                    --dataset $dataset \
                    --data_path $datadir/$i.label.jsonl
            fi
        } & done
fi

wait

# combine the tfidf files
python -u script/combineFile.py \
    --input_folder cache/$dataset/train_split \
    --output_folder cache/$dataset

echo -e "\033[34m[Shell] The preprocess of dataset $dataset has finished! \033[0m"

# run the script like:
# nohup python -u script/calw2sTFIDF.py --dataset qidian_1109_seq_winsize3 --data_path data/qidian_1109_seq_winsize3/train_split > preprocess_winsize3_parallel2.log 2>&1 &
# nohup python -u script/calw2sTFIDF.py --dataset qidian_1109_seq --data_path data/qidian_1109_seq/val.label.jsonl > multi_process_val.log 2>&1 &

# nohup bash PrepareDataset.sh qidian_1109_seq_winsize3 data/qidian_1109_seq_winsize3 multi > preprocess_winsize3_parallel.log 2>&1 &


# python script/combineFile.py \
#     --input_folder cache/qidian_1109_seq_winsize3/train_split \
#     --output_folder cache/qidian_1109_seq_winsize3