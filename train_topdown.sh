#!/bin/bash
dynet_seed=1

batch_size=16

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 3000,3000,500 \
    -T corpora/english/top_down/train.oracle \
    -d corpora/english/top_down/dev.oracle \
    -C corpora/english/dev.stripped \
    -t \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.filtered.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --batch_size $batch_size \
    $@
