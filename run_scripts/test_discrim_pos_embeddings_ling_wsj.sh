#!/bin/bash
model=$1
beam_size=$2
dev_or_test=$3
if [ -z $dev_or_test ]; then
    dev_or_test=dev
fi
build/nt-parser/nt-parser \
    --cnn-seed 1 \
    --cnn-mem 1000,0,500 \
    --model $model \
    -x \
    -T corpora/train.oracle \
    -p corpora/${dev_or_test}.oracle \
    -C corpora/${dev_or_test}.stripped \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --beam_size $beam_size \
    -D 0.2
