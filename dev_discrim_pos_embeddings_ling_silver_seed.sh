#!/bin/bash
build/nt-parser/nt-parser \
    --cnn-mem 3000,3000,4000 \
    -x \
    -T corpora/silver_train.oracle \
    -p corpora/silver_dev.oracle \
    -C corpora/dev.stripped \
    -P
    --pretrained_dim 100 \
    -w embeddings/GoogleNews-ptb_filtered-vectors-negative300.txt \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -m $1
