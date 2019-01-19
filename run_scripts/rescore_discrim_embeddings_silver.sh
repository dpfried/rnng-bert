#!/bin/bash

input_candidate_file=$1

#  **dev (iter=42075 epoch=52.8168)      llh=8250.16 ppl: 1.22833 f1: 92.26 err: 0.0203683       [1700 sents in 47705.8 ms]
discrim_model=${HOME}/snapshots/discrim_silver-1-to-1_2-4/models/ntparse_pos_pretrained_0_2_32_128_16_128-seed7-pid1802.params.bin
candidate_trees=${input_candidate_file}.rnng-discrim-embeddings-silver-trees
discrim_oracle=${input_candidate_file}.rnng-discrim-embeddings-silver-oracle-fmt
unindexed_output_file=${input_candidate_file}.unindexed-rnng-discrim-embeddings-silver-scores
output_file=${input_candidate_file}.rnng-discrim-embeddings-silver-scores

utils/cut-corpus.pl 3 $input_candidate_file > $candidate_trees

# todo: when using silver, won't be train.dictionary
python get_oracle.py corpora/silver_train.dictionary $candidate_trees > $discrim_oracle

build/nt-parser/nt-parser \
  --cnn-mem 20000,0,6000 \
  -x \
  -T corpora/silver_train.oracle \
  --gold_training_data corpora/silver_wsj-train.oracle \
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
