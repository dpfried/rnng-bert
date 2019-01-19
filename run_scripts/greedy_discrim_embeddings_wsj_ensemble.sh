#!/bin/bash

dataset=$1
combine_type=$2

shift
shift

build/nt-parser/nt-parser \
    --cnn-mem 15000,0,5000 \
    -x \
    -T corpora/train.oracle \
    -p corpora/${dataset}.oracle \
    -C corpora/${dataset}.stripped \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --combine_type $combine_type \
    --models "$@" 
    # -m $model_file
    # -p corpora/dev.oracle \
    # -C corpora/dev.stripped \
