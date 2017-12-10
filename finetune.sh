#!/bin/bash
dynet_seed=$1
method=$2
candidates=$3
model="snapshots/ntparse_pos_pretrained_0_2_32_128_16_128-seed1-pid2336.params.bin"
out_dir="sequence_level/finetune"
mkdir -p $out_dir
output_prefix="${out_dir}/1_method=${method}_candidates=${candidates}"

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
    --min_risk_training \
    --min_risk_method $method \
    --min_risk_candidates $candidates \
    -m $model \
    --model_output_file $output_prefix \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
