#!/bin/bash

source ../analysis_path.sh

SCRIPT_DIR="../../scripts"

STRIP_TOP=${SCRIPT_DIR}/strip_top.py
STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py
REPLACE_SYMBOLS=${SCRIPT_DIR}/replace_symbols.py

BERT_BASE_PATH="../../bert_models/uncased_L-12_H-768_A-12/"

DICTIONARY="../english/train.dictionary"

python get_ewt.py

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

for CATEGORY in answers email newsgroup reviews weblog
do
  for SPLIT in dev test
  do
    echo $CATEGORY $SPLIT

    STRIPPED=${CATEGORY}.${SPLIT}.stripped

    rm $STRIPPED
    ln -s ${ANALYSIS_DIR}/corpora/ewt/$STRIPPED $STRIPPED

    PROCESSED=${CATEGORY}.${SPLIT}.processed

    python $STRIP_FUNCTIONAL --remove_symbols EDITED META NML < $STRIPPED | python $REPLACE_SYMBOLS --map_from TOP --map_to FRAG > $PROCESSED

    # in_order discriminative
    python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH > in_order/${CATEGORY}.${SPLIT}.oracle
    # python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --collapse_unary > in_order/${CATEGORY}.collapse-unary.oracle

    # discriminative oracles
    # top_down discriminative
    python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH > top_down/${CATEGORY}.${SPLIT}.oracle
    # python $GET_ORACLE $DICTIONARY $PROCESSED --bert_model_dir $BERT_BASE_PATH --collapse_unary > top_down/${CATEGORY}.collapse-unary.oracle

    # top_down generative
    python $GET_ORACLE_GEN $DICTIONARY $PROCESSED > top_down/${CATEGORY}.${SPLIT}_gen.oracle

    echo
  done
done
