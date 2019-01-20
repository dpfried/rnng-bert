#!/bin/bash
dynet_seed=$1
shift

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 1500,1500,500 \
    -T corpora/english/in_order/train.oracle \
    -d corpora/english/in_order/dev.oracle \
    -C corpora/english/dev.stripped \
    --inorder \
    -t \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.filtered.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    $@
