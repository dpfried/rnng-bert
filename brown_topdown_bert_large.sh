#!/bin/bash
model_dir=$1
beam_size=$2
shift
shift

if [ -z $beam_size ]; then
    beam_size=1
fi

batch_size=8

build/nt-parser/nt-parser \
    --cnn-seed 1 \
    --cnn-mem 7000,2000,500 \
    --model_dir $model_dir \
    -T corpora/english/top_down/train.oracle \
    -p corpora/brown/top_down/Brown.goldtags.train.oracle \
    --bracketing_test_data corpora/brown/Brown.goldtags.train \
    --bert \
    --bert_large \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --beam_size $beam_size \
    --batch_size $batch_size \
    --eval_files_prefix brown_decodes/`basename $model_dir`-train-${beam_size}
