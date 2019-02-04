#!/bin/bash

dynet_seed=1
batch_size=16
bert_lr="2e-5"

prefix="topdown_collapse-unary_bert_large_bs=${batch_size}_lr=${bert_lr}"

output_dir="models/${prefix}"
mkdir $output_dir 2> /dev/null

echo $output_dir

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 4000,4000,500 \
    -T corpora/english/top_down/train.collapse-unary.oracle \
    -d corpora/english/top_down/dev.collapse-unary.oracle \
    -C corpora/english/dev.stripped \
    -t \
    --bert \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --batch_size $batch_size \
    --subbatch_max_tokens 500 \
    --eval_batch_size 8 \
    --bert_lr $bert_lr \
    --bert_large \
    --model_output_dir $output_dir \
    --collapse_unary \
    $@ \
    2>&1 | tee ${output_dir}/log.out
