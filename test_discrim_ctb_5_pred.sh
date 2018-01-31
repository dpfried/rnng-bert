#!/bin/bash
model=$1
dim=$2
beam_size=$3
dev_or_test=$4
if [ -z $dev_or_test ]; then
    dev_or_test=dev
fi
build/nt-parser/nt-parser \
    --cnn-seed 1 \
    --cnn-mem 1000,0,500 \
    --model $model \
    -x \
    -T ctb_5.1_corpora/train.pred.oracle \
    -p ctb_5.1_corpora/${dev_or_test}.pred.oracle \
    -C ctb_5.1_corpora/${dev_or_test}.pred.stripped \
    -P \
    --pretrained_dim 80 \
    -w embeddings/zzgiga.sskip.80.filtered_ctb_5.1.vectors \
    --lstm_input_dim $dim \
    --hidden_dim $dim \
    --beam_size $beam_size \
    -D 0.2
