#!/bin/bash
dynet_seed=$1
method=$2
dpe=$3
candidates=$4

if [ -z "$candidates" ]
then
  candidates=1
fi

out_dir="dynamic_oracle"
mkdir -p $out_dir 2> /dev/null
output_prefix="${out_dir}/${dynet_seed}_method=${method}_dpe=${dpe}_candidates=${candidates}_include-gold"

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 800,800,500 \
    -x \
    -T corpora/train.oracle \
    -d corpora/dev.oracle \
    -C corpora/dev.stripped \
    -t \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.filtered.vectors \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --model_output_file $output_prefix \
    --dynamic_exploration $method \
    --dynamic_exploration_candidates $candidates \
    --dynamic_exploration_probability $dpe \
    --dynamic_exploration_include_gold \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
