#!/bin/bash
build/nt-parser/nt-parser --cnn-mem 1700 -x -T corpora/train.oracle -p corpora/test.oracle -C corpora/test.stripped -P --lstm_input_dim 128 --hidden_dim 128 -m expts/ntparse_0_2_32_128_16_128-pid25393.params_101h 
