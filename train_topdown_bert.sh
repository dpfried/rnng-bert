#!/bin/bash
dynet_seed=$1
shift

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 1500,1500,500 \
    -T corpora/english/top_down/train.oracle \
    -d corpora/english/top_down/dev.oracle \
    -C corpora/english/dev.stripped \
    -t \
    --bert \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    $@
