#!/bin/bash
build/nt-parser/nt-parser --cnn-mem 1700 -x -T corpora/train.oracle -d corpora/dev.oracle -C corpora/dev.stripped -t --pretrained_dim 300 -w embeddings/GoogleNews-ptb_filtered-vectors-negative300.txt --lstm_input_dim 128 --hidden_dim 128 -D 0.2 > expts/discrim_no-pos_embed.stdout 2> expts/discrim_no-pos_embed.stderr
