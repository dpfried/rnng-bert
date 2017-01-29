#!/bin/bash
build/nt-parser/nt-parser --cnn-mem 1700 -x -T corpora/train.oracle -d corpora/dev.oracle -C corpora/dev.stripped -t --lstm_input_dim 128 --hidden_dim 128 -D 0.2 > expts/discrim_no-pos.stdout 2> expts/discrim_no-pos.stderr
