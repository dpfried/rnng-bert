#!/bin/bash
dynet_seed=$1
method=$2
dpe=$3

out_dir="dynamic_oracle"
mkdir -p $out_dir
output_prefix="${out_dir}/${dynet_seed}_method=${method}_dpe=${dpe}"

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 2000 \
    -x \
    -T corpora/train.oracle \
    -d corpora/dev.oracle \
    -C corpora/dev.stripped \
    -t \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --model_output_file $output_prefix \
    --dynamic_exploration $method \
    --dynamic_exploration_probability $dpe \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
