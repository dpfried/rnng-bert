#!/bin/bash
dynet_seed=$1
optimizer=$2

if [ -z "$optimizer" ]
then
    optimizer="sgd"
fi

out_dir="sequence_level"
mkdir -p $out_dir
output_prefix="${out_dir}/${dynet_seed}_method=max_margin_opt=${optimizer}"

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 1500,1500,500 \
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
    --max_margin_training \
    --unnormalized \
    --model_output_file $output_prefix \
    --optimizer $optimizer \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
