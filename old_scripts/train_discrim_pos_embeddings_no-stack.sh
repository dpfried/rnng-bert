#!/bin/bash
output_path=$1
build/nt-parser/nt-parser --cnn-mem 1700 -x -T corpora/train.oracle -d corpora/dev.oracle -C corpora/dev.stripped -t --pretrained_dim 300 -w embeddings/GoogleNews-ptb_filtered-vectors-negative300.txt -P --lstm_input_dim 128 --hidden_dim 128 -D 0.2 --no_stack > ${output_path}.stdout 2> ${output_path}.stderr
