#!/bin/bash
model_dir=$1
dev_or_test=$2
beam_size=$3
shift
shift
shift

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
    -T corpora/english/top_down/train.collapse-unary.oracle \
    -p corpora/english/top_down/${dev_or_test}.collapse-unary.oracle \
    --bracketing_test_data corpora/english/${dev_or_test}.stripped \
    --bert \
    --bert_large \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --beam_size $beam_size \
    --batch_size $batch_size \
    --eval_files_prefix decodes/`basename $model_dir`-${dev_or_test}-${beam_size} \
    --collapse_unary \
    $@
