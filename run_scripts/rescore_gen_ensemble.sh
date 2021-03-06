#!/bin/bash

input_candidate_file=$1

#gen_model=${HOME}/snapshots/gen_wsj_1-31/models/ntparse_gen_D0.3_2_256_256_16_256-seed5-pid1484.params.bin

candidate_trees=${input_candidate_file}.rnng-gen-wsj-ensemble-sum-trees

candidate_trees_unked=${input_candidate_file}.rnng-gen-wsj-ensemble-sum-trees-unked

score_output_file=${input_candidate_file}.raw-rnng-gen-wsj-ensemble-sum-scores
output_file=${input_candidate_file}.rnng-gen-wsj-ensemble-sum-scores

# TODO: modify this for silver
train_dictionary=corpora/train.dictionary

utils/cut-corpus.pl 3 $input_candidate_file > $candidate_trees

python add_dev_unk.py $train_dictionary $candidate_trees > $candidate_trees_unked

build/nt-parser/nt-parser-gen \
  --cnn-mem 35000,0,4000 \
  -x \
  -T corpora/train_gen.oracle \
  --clusters clusters-train-berk.txt \
  --input_dim 256 \
  --lstm_input_dim 256 \
  --hidden_dim 256 \
  -p $candidate_trees_unked \
  --models ${HOME}/snapshots/gen_wsj_1-31/models/* \
  --combine_type sum \
  > $score_output_file

python add_gen_scores.py $input_candidate_file $score_output_file > $output_file
