#!/bin/bash

source ../analysis_path.sh

SCRIPT_DIR="../../scripts"

STRIP_ROOT=${SCRIPT_DIR}/strip_root.py
STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py

BERT_BASE_PATH="../../bert_models/uncased_L-12_H-768_A-12/"
BERT_LARGE_PATH="../../bert_models/uncased_L-24_H-1024_A-16/"

DICTIONARY=train.dictionary

ln -s ../english/train.dictionary $DICTIONARY

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

for SPLIT in cf cg ck cl cm cn cp cr train test test40
do
  STRIPPED=${SPLIT}.pred.stripped
  PROCESSED=${SPLIT}.pred.processed

  ln -s $ANALYSIS_DIR/corpora/brown/${SPLIT}.gold.stripped ${SPLIT}.gold.stripped

  rm $STRIPPED
  ln -s $ANALYSIS_DIR/corpora/brown/$STRIPPED $STRIPPED

  python $STRIP_FUNCTIONAL < $STRIPPED \
    --remove_symbols ADV AUX EDITED NEG TYPO UH \
    --remove_root_must_have TOP \
    --root_removed_replacement X \
    > $PROCESSED

  # in_order discriminative
  python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH > in_order/${SPLIT}.pred.oracle
  # python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --collapse_unary > in_order/${SPLIT}.collapse-unary.pred.oracle

  # discriminative oracles
  # top_down discriminative
  python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH > top_down/${SPLIT}.pred.oracle
  # python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --collapse_unary > top_down/${SPLIT}.collapse-unary.pred.oracle

  # top_down generative
  # python $GET_ORACLE_GEN $DICTIONARY $PROCESSED > top_down/${SPLIT}_gen.pred.oracle
done
