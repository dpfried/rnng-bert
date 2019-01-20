#!/bin/bash
dynet_seed=1

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 1500,1500,500 \
    -T corpora/english/in_order/train.oracle \
    -d corpora/english/in_order/dev.oracle \
    -C corpora/english/dev.stripped \
    --inorder \
    -t \
    --bert \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --batch_size 8 \
    $@
