#!/bin/bash

# {dev, test}
dataset=$1
beam_size=$2

# epoch=52.82, dev f1=92.26
model_file=${HOME}/snapshots/discrim_silver-1-to-1_2-4/models/ntparse_pos_pretrained_0_2_32_128_16_128-seed7-pid1802.params.bin

name=rnng-discrim-embeddings-silver

out_prefix=${dataset}_${name}_beam=${beam_size}.candidates

output_candidate_file=${HOME}/candidates/${out_prefix}

build/nt-parser/nt-parser \
    --cnn-mem 3000,0,4000 \
    -x \
    -T corpora/silver_train.oracle \
    -p corpora/silver_${dataset}.oracle \
    -C corpora/${dataset}.stripped \
    --gold_training_data corpora/silver_wsj-train.oracle \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --beam_size $beam_size \
    --output_beam_as_samples \
    --ptb_output_file $output_candidate_file \
    -m $model_file
