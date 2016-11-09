#!/bin/bash
build/nt-parser/nt-parser --cnn-mem 1700 -x -T corpora/train.oracle -p corpora/test.oracle -C corpora/test.stripped --pretrained_dim [dimension of pre-trained word embedding] -w [pre-trained word embedding] --lstm_input_dim 128 --hidden_dim 128 -m [parameter file] > output.txt 
