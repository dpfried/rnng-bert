#!/bin/bash

input_candidate_file=$1
discrim_model=$2
candidate_trees=${input_candidate_file}.rnng-discrim-trees
discrim_oracle=${input_candidate_file}.rnng-discrim-oracle-fmt
unindexed_output_file=${input_candidate_file}.unindexed-rnng-discrim-scores
output_file=${input_candidate_file}.rnng-discrim-scores

utils/cut-corpus.pl 3 $input_candidate_file > $candidate_trees
python get_oracle.py corpora/train.dictionary $candidate_trees > $discrim_oracle

build/nt-parser/nt-parser \
  --cnn-mem 1700 \
  -x \
  -T corpora/train.oracle \
  -p $discrim_oracle \
  -C $candidate_trees \
  --pretrained_dim 100 \
  -w embeddings/sskip.100.vectors \
  -P \
  --lstm_input_dim 128 \
  --hidden_dim 128 \
  -m $discrim_model \
  --samples 0 \
  --samples_include_gold \
  --ptb_output_file $unindexed_output_file \
  > /dev/null

python reindex.py $input_candidate_file $unindexed_output_file > $output_file
