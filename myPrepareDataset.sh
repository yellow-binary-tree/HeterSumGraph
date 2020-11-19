#!/bin/bash

# run the script like:

dataset="$1"
task="$2"

set -e

if [ ! -n "$dataset" ]; then
    echo "lose the dataset name!"
    exit
fi

if [ ! -n "$task" ]; then
    task=HSG
fi

if [ $task == 'HSG' ]; then
    doc_max_timesteps=70
elif [ $task == 'HDSG3' ]; then
    doc_max_timesteps=150
elif [ $task == 'HDSG5' ]; then
    doc_max_timesteps=230
elif [ $task == 'HDSG7' ]; then
    doc_max_timesteps=310
fi

num_proc=8

# {
#     echo -e "\033[34m[Shell] Get low tfidf words from training set! \033[0m"
#     python -u script/myLowTFIDFWords.py --dataset $dataset
# } &
# echo -e "\033[34m[Shell] Get word2sent edge feature! \033[0m"
# for i in `seq 1 $num_proc`
# do {
#     python -u script/myCalw2sTFIDF.py --dataset $dataset --num_proc $num_proc --no_proc $i
# } & done

# if [ "$task" != "HSG" ]; 
# then {
#     echo -e "\033[34m[Shell] Get word2doc edge feature! \033[0m"
#         python -u script/myCalw2dTFIDF.py --dataset $dataset 
# } & fi
# {
#     echo -e "\033[34m[Shell] Create Vocabulary! \033[0m"
#     python -u script/myCreateVoc.py --dataset $dataset
# }

wait

echo -e "\033[34m[Shell] Building Graphs! \033[0m"
for i in `seq 1 $num_proc`
do {
    python -u script/myCreateGraph.py \
        --dataset $dataset --doc_max_timesteps $doc_max_timesteps --sent_max_len 50 --model $task \
        --num_proc $num_proc --no_proc $i
} & done
