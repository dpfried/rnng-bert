#!/bin/bash
build/nt-parser/nt-parser-gen --cnn-mem 3000 -x -T corpora/train_gen.oracle -d corpora/dev_gen.oracle -C corpora/dev.stripped --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -m expts/ntparse_gen_D0.3_2_256_256_16_256-pid4187.params_20.3379 --greedy_decode_dev "$@"
