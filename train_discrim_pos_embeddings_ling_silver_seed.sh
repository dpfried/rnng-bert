#!/bin/bash
dynet_seed=$1
output_prefix=expts/discrim_silver_embeddings_${dynet_seed}
build/nt-parser/nt-parser --cnn-seed $dynet_seed --cnn-mem 20000 -x -T corpora/silver_train.oracle -d corpora/silver_dev.oracle -C corpora/dev.stripped -t --pretrained_dim 100 -P -w embeddings/sskip.100.vectors --lstm_input_dim 128 --hidden_dim 128 -D 0.2 > ${output_prefix}.stdout 2> ${output_prefix}.stderr
