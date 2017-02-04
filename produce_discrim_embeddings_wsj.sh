#!/bin/bash

# {dev, test}
dataset=$1
output_candidate_file=$2
model_file=$3
beam_size=$4

build/nt-parser/nt-parser \
    --cnn-mem 1700 \
    -x \
    -T corpora/train.oracle \
    -p corpora/${dataset}.oracle \
    -C corpora/${dataset}.stripped \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --beam_size $beam_size \
    --output_beam_as_samples \
    --ptb_output_file $output_candidate_file \
    -m $model_file
