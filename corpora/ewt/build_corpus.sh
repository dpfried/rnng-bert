#!/bin/bash

source ../analysis_path.sh

SCRIPT_DIR="../../scripts"

STRIP_ROOT=${SCRIPT_DIR}/strip_root.py
STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py
REPLACE_SYMBOLS=${SCRIPT_DIR}/replace_symbols.py

BERT_BASE_PATH="../../bert_models/uncased_L-12_H-768_A-12/"

DICTIONARY="../english/train.dictionary"

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

for CATEGORY in answers email newsgroup reviews weblog
do
  for SPLIT in dev test
  do
    echo $CATEGORY $SPLIT

    STRIPPED=${CATEGORY}.${SPLIT}.pred.stripped

    rm $STRIPPED
    ln -s ${ANALYSIS_DIR}/corpora/ewt/$STRIPPED $STRIPPED

    ln -s ${ANALYSIS_DIR}/corpora/ewt/${CATEGORY}.${SPLIT}.gold.stripped ${CATEGORY}.${SPLIT}.gold.stripped

    PROCESSED=${CATEGORY}.${SPLIT}.pred.processed

    cat $STRIPPED | \
      python $STRIP_FUNCTIONAL \
      --remove_symbols EDITED META NML \
      --remove_root TOP \
      --root_removed_replacement X \
      > $PROCESSED

    # in_order discriminative
    python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH > in_order/${CATEGORY}.${SPLIT}.pred.oracle
    # python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --collapse_unary > in_order/${CATEGORY}.collapse-unary.pred.oracle

    # discriminative oracles
    # top_down discriminative
    python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH > top_down/${CATEGORY}.${SPLIT}.pred.oracle
    # python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --collapse_unary > top_down/${CATEGORY}.collapse-unary.pred.oracle

    # top_down generative
    python $GET_ORACLE_GEN $DICTIONARY $PROCESSED > top_down/${CATEGORY}.${SPLIT}_gen.pred.oracle

    echo
  done
done
