#!/bin/bash

model_dir=$1
decodes_oracle=$2

basename="${decodes_oracle%.topdown.oracle}"

decodes_and_scores=${basename}.topdown.unindexed_scores
decodes_bracketed=${basename}.topdown.bracketed
decodes_and_scores_reindexed=${basename}.topdown.scores

echo $decodes_and_scores
echo $decodes_bracketed

build/nt-parser/nt-parser \
  --cnn-mem 2000,500,500 \
  --cnn-seed 1 \
  -T corpora/english/top_down/train.oracle \
  -p $decodes_oracle \
  --model_dir $model_dir \
  --bert \
  --bert_large \
  --lstm_input_dim 128 \
  --hidden_dim 128 \
  --batch_size 8 \
  --samples 0 \
  --samples_include_gold \
  --ptb_output_file $decodes_and_scores \
  > /dev/null

# get just the trees
utils/cut-corpus.pl 3 $decodes_and_scores > $decodes_bracketed

# remove whitespace at line beginning, and then ignore whitespace changes
# echo "diff check: " $(diff -bB <(sed 's/^[ \t]*//' $decodes_input) <(sed 's/^[ \t]*//' $decodes_bracketed))

python scripts/reindex.py $basename $decodes_and_scores > $decodes_and_scores_reindexed
