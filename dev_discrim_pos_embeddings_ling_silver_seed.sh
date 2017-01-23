#!/bin/bash
build/nt-parser/nt-parser \
    --cnn-mem 3000,0,4000 \
    -x \
    -T corpora/silver_train.oracle \
    -p corpora/silver_dev.oracle \
    -C corpora/dev.stripped \
    --gold_training_data corpora/silver_wsj-train.oracle \
    -P
    --pretrained_dim 100 \
    -w embeddings/sskip.100.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -m $1
