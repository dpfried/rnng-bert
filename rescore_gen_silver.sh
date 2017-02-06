#!/bin/bash

input_candidate_file=$1

gen_model=${HOME}/snapshots/gen_silver-1-to-1_2-4/models/ntparse_gen_D0.3_2_256_256_16_256-seed1-pid1871.params.bin

candidate_trees=${input_candidate_file}.rnng-gen-silver-trees

candidate_trees_unked=${input_candidate_file}.rnng-gen-silver-trees-unked

score_output_file=${input_candidate_file}.raw-rnng-gen-silver-scores
output_file=${input_candidate_file}.rnng-gen-silver-scores

# TODO: modify this for silver
train_dictionary=corpora/silver_train.dictionary

utils/cut-corpus.pl 3 $input_candidate_file > $candidate_trees

python add_dev_unk.py $train_dictionary $candidate_trees > $candidate_trees_unked

build/nt-parser/nt-parser-gen \
    --cnn-mem 25000,0,8000 \
  -x \
  -T corpora/silver_train_gen.oracle \
  --gold_training_data corpora/silver_wsj-train_gen.oracle \
  --clusters clusters-silver-train-berk.txt \
  --input_dim 256 \
  --lstm_input_dim 256 \
  --hidden_dim 256 \
  -p $candidate_trees_unked \
  -m $gen_model \
  > $score_output_file

python add_gen_scores.py $input_candidate_file $score_output_file > $output_file
