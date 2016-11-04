#!/bin/bash
build/nt-parser/nt-parser-gen -x -T corpora/train.oracle -d corpora/dev.oracle -t --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -D 0.3 > expts/gen_log.txt
