#!/bin/bash

input_candidate_file=$1
discrim_model=$HOME/snapshots/discrim_wsj_embeddings_1-31/models/ntparse_pos_pretrained_0_2_32_128_16_128-seed3-pid7347.params.bin
candidate_trees=${input_candidate_file}.rnng-discrim-embeddings-wsj-trees
discrim_oracle=${input_candidate_file}.rnng-discrim-embeddings-wsj-oracle-fmt
unindexed_output_file=${input_candidate_file}.unindexed-rnng-discrim-embeddings-wsj-scores
output_file=${input_candidate_file}.rnng-discrim-embeddings-wsj-scores


utils/cut-corpus.pl 3 $input_candidate_file > $candidate_trees

# todo: when using silver, won't be train.dictionary
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
