#!/bin/bash
model_dir=$1
beam_size=$2
block_num=$3
shift
shift
shift

if [ -z $beam_size ]; then
    beam_size=1
fi

block_count=4
if [ -z $block_num ]; then
    dev_or_test=dev
    block_count=0
    block_num=0
fi

batch_size=8

mkdir brown_k_best 2> /dev/null

build/nt-parser/nt-parser \
    --cnn-seed 1 \
    --cnn-mem 10000,2000,500 \
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
    --output_beam_as_samples \
    --ptb_output_file brown_k_best/`basename $model_dir`-train-${beam_size}.txt \
    --block_num $block_num \
    --block_count $block_count \
    $@
