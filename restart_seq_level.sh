#!/bin/bash
dynet_seed=$1
set_iter=$2
method=reinforce
candidates=10
optimizer=sgd
batch_size=1
dim=$3

if [ -z "$optimizer" ]
then
    optimizer="sgd"
fi

out_dir="sequence_level"
mkdir -p $out_dir


output_prefix="${out_dir}/${dynet_seed}_method=${method}_candidates=${candidates}_opt=${optimizer}_include-gold"

if [ -z "$batch_size" ]
then
  batch_size=1
else
  output_prefix="${output_prefix}_batch-size=${batch_size}"
fi

if [ -z "$dim" ]
then
  dim=128
else
  output_prefix="${output_prefix}_dim=${dim}"
fi

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 2000,2000,500 \
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
    --min_risk_include_gold \
    --optimizer $optimizer \
    --batch_size $batch_size \
    --set_iter $set_iter \
    -m ${output_prefix}.bin \
    --model_output_file $output_prefix \
    >> ${output_prefix}.stdout \
    2>> ${output_prefix}.stderr
