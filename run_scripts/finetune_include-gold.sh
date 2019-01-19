#!/bin/bash
dynet_seed=$1
method=$2
candidates=$3
optimizer=$4
batch_size=$5
sgd_e0=$6

if [ -z "$optimizer" ]
then
    optimizer="sgd"
fi

model="snapshots/ntparse_pos_pretrained_0_2_32_128_16_128-seed3-pid7347.params.bin"
out_dir="sequence_level/finetune"
mkdir -p $out_dir

output_prefix="${out_dir}/1_method=${method}_candidates=${candidates}_opt=${optimizer}_include-gold"
if [ -z "$batch_size" ]
then
  batch_size=1
else
  output_prefix="${output_prefix}_batch-size=${batch_size}"
fi

if [ -z "$sgd_e0" ]
then
  sgd_e0=0.1
else
  output_prefix="${output_prefix}_sgd-e0=${sgd_e0}"
fi

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 3000,3000,500 \
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
    --min_risk_training \
    --min_risk_method $method \
    --min_risk_candidates $candidates \
    --min_risk_include_gold \
    -m $model \
    --model_output_file $output_prefix \
    --optimizer $optimizer \
    --sgd_e0 $sgd_e0 \
    --batch_size $batch_size \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
