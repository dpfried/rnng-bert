#!/bin/bash
dynet_seed=$1
method=$2
candidates=$3
optimizer=$4
dim=$5

if [ -z "$4" ]
then
    optimizer="sgd"
fi

out_dir="sequence_level"
mkdir -p $out_dir
output_prefix="${out_dir}/${dynet_seed}_method=${method}_candidates=${candidates}_opt=${optimizer}"

if [ -z "$dim" ]
then
  dim=128
else
  output_prefix="${output_prefix}_dim=${dim}"
fi

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 2000,2000,1000 \
    -x \
    -T corpora/train.oracle \
    -d corpora/dev.oracle \
    -C corpora/dev.stripped \
    -t \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.vectors \
    --lstm_input_dim $dim \
    --hidden_dim $dim \
    -D 0.2 \
    --min_risk_training \
    --min_risk_method $method \
    --min_risk_candidates $candidates \
    --model_output_file $output_prefix \
    --optimizer $optimizer \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
