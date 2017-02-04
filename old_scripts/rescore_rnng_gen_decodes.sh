#!/bin/bash

decode_file=$1
discrim_oracle=${decode_file}.oracle_fmt
decodes_and_discrim_scores=${decode_file}.rnng-discrim-scores
decode_trees_rnng_format=${decode_file}.rnng-format
gen_scores=${decode_file}.rnng-gen-scores
discrim_model="../rnng_adhi/expts/ntparse_pos_0_2_32_128_16_128-pid28571.params_91.94"

python get_oracle.py corpora/train.dictionary $decode_file > $discrim_oracle

build/nt-parser/nt-parser -x -T corpora/train.oracle -p $discrim_oracle -C $decode_file --pretrained_dim 100 -w embeddings/sskip.100.filtered.vectors -P --lstm_input_dim 128 --hidden_dim 128 -m $discrim_model --samples 0 --samples_include_gold > $decodes_and_discrim_scores

utils/cut-corpus.pl 3 $decode_and_discrim_scores > $decode_trees_rnng_format
