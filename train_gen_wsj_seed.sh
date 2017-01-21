#!/bin/bash
dynet_seed=$1
output_prefix=expts/gen_wsj_${dynet_seed}
build/nt-parser/nt-parser-gen --cnn-seed ${dynet_seed} --cnn-memory 2000 -x -T corpora/train_gen.oracle -d corpora/dev_gen.oracle -t --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -D 0.3 > ${output_prefix}.stdout 2> ${output_prefix}.stderr
