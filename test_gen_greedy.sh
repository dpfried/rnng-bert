#!/bin/bash
build/nt-parser/nt-parser-gen -x -T corpora/train_gen.oracle -p corpora/test_gen.oracle -C corpora/test.stripped --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -m [parameter file] --greedy_decode_in_test 
