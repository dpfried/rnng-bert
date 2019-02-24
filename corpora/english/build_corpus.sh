#!/bin/bash

source ../analysis_path.sh

SCRIPT_DIR="../../scripts"

STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py

BERT_BASE_PATH="../../bert_models/uncased_L-12_H-768_A-12/"
BERT_LARGE_PATH="../../bert_models/uncased_L-24_H-1024_A-16/"

DICTIONARY="train.dictionary"

for SPLIT in train dev test
do
  STRIPPED=${SPLIT}.stripped
  rm $STRIPPED
  ln -s ${ANALYSIS_DIR}/corpora/wsj_ch/$STRIPPED $STRIPPED
  PROCESSED=${SPLIT}.processed

  python $STRIP_FUNCTIONAL --remove_root TOP < $STRIPPED > $PROCESSED
done

python $GET_DICTIONARY train.processed > $DICTIONARY

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

for SPLIT in train dev test
do
  PROCESSED=${SPLIT}.processed
  # discriminative oracles

  # top_down discriminative
  python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH > top_down/${SPLIT}.oracle
  # python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --collapse_unary > top_down/${SPLIT}.collapse-unary.oracle
  # python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --reverse_trees > top_down/${SPLIT}.reverse-trees.oracle

  # the bert_large oracle files are identical because Base and Large, for English uncased at least, have the same vocab.txt
  #python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --bert_model_dir $BERT_LARGE_PATH > top_down/${SPLIT}_bert_large.oracle

  # top_down generative
  python $GET_ORACLE_GEN $DICTIONARY $PROCESSED > top_down/${SPLIT}_gen.oracle

  # in_order discriminative
  python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH > in_order/${SPLIT}.oracle
  # python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --collapse_unary > in_order/${SPLIT}.collapse-unary.oracle
  # python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --reverse_trees > in_order/${SPLIT}.reverse-trees.oracle

  # the bert_large oracle files are identical because Base and Large, for English uncased at least, have the same vocab.txt
  #python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --bert_model_dir $BERT_LARGE_PATH > in_order/${SPLIT}_bert_large.oracle
done
