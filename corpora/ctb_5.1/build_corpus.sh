#!/bin/bash

TRAIN="train.gold.original"
DEV="dev.gold.original"
TEST="test.gold.original"

SCRIPT_DIR="../../scripts"

REMOVE_TRACES=${SCRIPT_DIR}/remove_traces.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py

BERT_PATH="../../bert_models/chinese_L-12_H-768_A-12/"

python ${REMOVE_TRACES} < $TRAIN > train.gold.stripped
python ${REMOVE_TRACES} < $DEV > dev.gold.stripped
python ${REMOVE_TRACES} < $TEST > test.gold.stripped

DICTIONARY="train.dictionary"

python $GET_DICTIONARY train.gold.stripped > $DICTIONARY

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

# discriminative oracles
for SPLIT in train.gold dev.gold test.gold
do
  # top_down discriminative
  python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH > top_down/${SPLIT}.oracle
  #python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH --collapse_unary > top_down/${SPLIT}.collapse-unary.oracle
  #python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH --reverse_trees > top_down/${SPLIT}.reverse-trees.oracle

  # top_down generative
  python $GET_ORACLE_GEN $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking > top_down/${SPLIT}_gen.oracle

  # in_order discriminative
  python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH > in_order/${SPLIT}.oracle
  #python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH --collapse_unary > in_order/${SPLIT}.collapse-unary.oracle
  #python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH --reverse_trees > in_order/${SPLIT}.reverse-trees.oracle

done
