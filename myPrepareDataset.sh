#!/bin/bash

# run the script like:
# nohup bash myPrepareDataset.sh winsize3_bow_cut_sennum20 HDSG3 20 > HDSGbc3_prepeocess_sennum20.log  2>&1 &
# nohup bash myPrepareDataset.sh winsize5_bow_cut_sennum20 HDSG5 20 > HDSGbc5_prepeocess_sennum20.log  2>&1 &

dataset="$1"
task="$2"
winsize_chap_sents="$3"

set -e

if [ ! -n "$dataset" ]; then
    echo "lose the dataset name!"
    exit
fi

if [ ! -n "$task" ]; then
    task=HSG
fi

if [ ! -n "$winsize_chap_sents" ]; then
    winsize_chap_sents=10
fi

if [ $task == 'HSG' ]; then
    doc_max_timesteps=30
elif [ $task == 'HDSG3' ]; then
    doc_max_timesteps=$(( 30+2*$winsize_chap_sents ))
elif [ $task == 'HDSG5' ]; then
    doc_max_timesteps=$(( 30+4*$winsize_chap_sents ))
elif [ $task == 'HDSG7' ]; then
    doc_max_timesteps=$(( 30+6*$winsize_chap_sents ))
fi

echo "dataset: $dataset, task: $task, doc_max_timesteps: $doc_max_timesteps"

num_proc=8

{
    echo -e "\033[34m[Shell] Get low tfidf words from training set! \033[0m"
    python -u script/myLowTFIDFWords.py --dataset $dataset
} &
echo -e "\033[34m[Shell] Get word2sent edge feature! \033[0m"
for i in `seq 1 $num_proc`
do {
    python -u script/myCalw2sTFIDF.py --dataset $dataset --num_proc $num_proc --no_proc $i
} & done

if [ "$task" != "HSG" ]; 
then {
    echo -e "\033[34m[Shell] Get word2doc edge feature! \033[0m"
        python -u script/myCalw2dTFIDF.py --dataset $dataset 
} & fi
{
    echo -e "\033[34m[Shell] Create Vocabulary! \033[0m"
    python -u script/myCreateVoc.py --dataset $dataset
} &
{
    echo -e "\033[34m[Shell] Create Test Vocabulary! \033[0m"
    python -u script/myCreateTestVoc.py --dataset $dataset
}

wait

echo -e "\033[34m[Shell] Building Graphs! \033[0m"
for i in `seq 1 $num_proc`
do {
    python -u script/myCreateGraph.py \
        --dataset $dataset --doc_max_timesteps $doc_max_timesteps --sent_max_len 50 --model $task \
        --num_proc $num_proc --no_proc $i
} & done

# build test graph
{
    python -u script/myCreateTestGraph.py \
        --dataset $dataset --doc_max_timesteps $doc_max_timesteps --sent_max_len 50 --model $task \
        --num_proc 1 --no_proc 1
} 

wait
echo -e "\033[34m[Shell] Preprocess Finished! \033[0m"
