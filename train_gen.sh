#!/bin/bash
build/nt-parser/nt-parser-gen -x -T corpora/train_gen.oracle -d corpora/dev_gen.oracle -t --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -D 0.3 > expts/gen.stdout 2> expts/gen.stderr
