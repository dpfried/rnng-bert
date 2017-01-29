#!/bin/bash

build/nt-parser/nt-parser --cnn-mem 1700 -x -T corpora/train.oracle -p corpora/test.oracle -C corpora/test.stripped -P --lstm_input_dim 128 --hidden_dim 128 -m ntparse_pos_0_2_32_128_16_128-pid4314.params --alpha 0.8 -s 100 > expts/discrim_test.samples
