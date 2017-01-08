#!/bin/bash
build/nt-parser/nt-parser-gen --cnn_seed 3930287672 -x -T corpora/train_gen.oracle -d corpora/dev_gen.oracle -t --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -D 0.3 --no_history --no_buffer --epoch_serialization_interval 10 > expts/gen_no-history_no-buffer2.stdout 2> expts/gen_no-history_no-buffer2.stderr
