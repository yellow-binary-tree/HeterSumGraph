# testall.sh
# test all the saved module in an experiment

# run the script like:
# nohup bash testall.sh HDSG 20201126_204634 winsize3_bow_cut 3 1 > HDSGbc3_testall3.log 2>&1 &
# nohup bash testall.sh HSG 20201126_204639 winsize1_random_cut 1 1 > HSGrc1_testall3.log 2>&1 &

model=$1
save_dir=$2
dataset=$3
winsize=$4
gpu=$5

batch_size=16

if [ $winsize == 1 ]; then
    doc_max_timesteps=30
elif [ $winsize == 3 ]; then
    doc_max_timesteps=50
elif [ $winsize == 5 ]; then
    doc_max_timesteps=70
elif [ $winsize == 7 ]; then
    doc_max_timesteps=90
fi

echo 'testall.sh: test '$model $dataset $save_dir $winsize $gpu
model_filenames=$(ls save/$save_dir/train)
# decode_filenames=(iter_53465 iter_56610 iter_59755 iter_62900 iter_66045 iter_69190 iter_72335 iter_75480 iter_78625)
decode_filenames=(iter_47175 iter_50320)

for model_filename in ${decode_filenames[@]}
do
    echo "testall.sh testing model train$model_filename"
    python -u evaluation.py \
        --model $model --exp_name myHeterSumGraph_test_${model}_${dataset} \
        --data_dir /share/wangyq/project/HeterSumGraph/data/$dataset \
        --cache_dir /share/wangyq/project/HeterSumGraph/cache/$dataset \
        --save_root save/$save_dir --log_root log/ --test_model train$model_filename \
        --embedding_path Tencent_AILab_ChineseEmbedding_200w.txt --word_emb_dim 200 \
        --vocab_size 100000 --batch_size $batch_size \
        --sent_max_len 50 --doc_max_timesteps $doc_max_timesteps -m 5 \
        --save_label --cuda --gpu $gpu
done
