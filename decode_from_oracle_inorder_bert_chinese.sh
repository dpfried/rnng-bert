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

mkdir decodes_ctb 2> /dev/null

build/nt-parser/nt-parser \
    --cnn-seed 1 \
    --cnn-mem 3000,3000,500 \
    --model_dir $model_dir \
    -p corpora/ctb_5.1/in_order/${dev_or_test}.gold.oracle \
    --bracketing_test_data corpora/ctb_5.1/${dev_or_test}.gold.stripped
    --max_unary 5 \
    --inorder \
    --bert \
    --bert_model_dir bert_models/chinese_L-12_H-768_A-12 \
    --bert_graph_path bert_models/chinese_L-12_H-768_A-12_graph.pb \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --beam_size $beam_size \
    --batch_size $batch_size \
    --text_format \
    $@
