#!/bin/bash

# {dev, test}
dataset=$1
beam_size=$2

# epoch=92.9554, 92.09 dev f1
model_file=$HOME/snapshots/discrim_wsj_embeddings_1-31/models/ntparse_pos_pretrained_0_2_32_128_16_128-seed3-pid7347.params.bin

name=rnng-discrim-embeddings-wsj

out_prefix=${dataset}_${name}_beam=${beam_size}.candidates

output_candidate_file=${HOME}/candidates/${out_prefix}

build/nt-parser/nt-parser \
    --cnn-mem 2000,0,500 \
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
