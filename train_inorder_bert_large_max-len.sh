#!/bin/bash

dynet_seed=1
batch_size=8
bert_lr="2e-5"

max_sentence_length_train=40

prefix="inorder_bert_large_bs=${batch_size}_lr=${bert_lr}_mslt=${max_sentence_length_train}"

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 3000,3000,500 \
    -T corpora/english/in_order/train.oracle \
    -d corpora/english/in_order/dev.oracle \
    -C corpora/english/dev.stripped \
    --inorder \
    -t \
    --bert \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --batch_size $batch_size \
    --bert_lr $bert_lr \
    --bert_large \
    --model_output_dir models/${prefix} \
    --max_sentence_length_train $max_sentence_length_train \
    $@ \
    2>&1 | tee logs/${prefix}.out
