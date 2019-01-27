#!/bin/bash
dynet_seed=1

batch_size=16
bert_lr="2e-5"

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 1000,1000,500 \
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
    --batch_size $batch_size \
    $@
