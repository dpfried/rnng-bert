#!/bin/bash
build/nt-parser/nt-parser-gen --cnn-mem 3000 -x -T corpora/train_gen.oracle -d corpora/dev_gen.oracle -C corpora/dev.stripped --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -m ntparse_gen_D0.3_2_256_256_16_256-pid4187.params --greedy_decode_dev "$@"
