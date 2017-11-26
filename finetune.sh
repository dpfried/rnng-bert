#!/bin/bash
dynet_seed=$1
model="snapshots/ntparse_pos_pretrained_0_2_32_128_16_128-seed1-pid2336.params.bin"
output_prefix="finetune/1"
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
    --min_risk_samples 10 \
    -m $model \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
