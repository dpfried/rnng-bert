#!/bin/bash
model_dir=$1
beam_size=$2
dev_or_test=$3

if [ -z $beam_size ]; then
    beam_size=1
fi

if [ -z $dev_or_test ]; then
    dev_or_test=dev
fi

batch_size=8

build/nt-parser/nt-parser \
    --cnn-seed 1 \
    --cnn-mem 1500,1500,500 \
    --model_dir $model_dir \
    -T corpora/english/in_order/train.oracle \
    -p corpora/english/in_order/${dev_or_test}.oracle \
    -C corpora/english/${dev_or_test}.stripped \
    --inorder \
    --bert \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --beam_size $beam_size \
    --batch_size $batch_size \
    $@
