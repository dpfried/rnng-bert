#!/bin/bash

SCRIPT_DIR="../../scripts"

STRIP_TOP=${SCRIPT_DIR}/strip_top.py
STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py

BERT_BASE_PATH="../../bert_models/uncased_L-12_H-768_A-12/"
BERT_LARGE_PATH="../../bert_models/uncased_L-24_H-1024_A-16/"


DICTIONARY="ptb.dictionary"

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

for base in dev.trees.nos1s; do
  STRIPPED=${base}.stripped

  python ${STRIP_TOP} < $base | python $STRIP_FUNCTIONAL > $STRIPPED

  # in_order discriminative
  python $GET_ORACLE --in_order $DICTIONARY $STRIPPED --bert_model_dir $BERT_BASE_PATH \
    > in_order/`basename $base`.ptb_dictionary.oracle

  # top_down discriminative
  python $GET_ORACLE $DICTIONARY $STRIPPED --bert_model_dir $BERT_BASE_PATH \
    > top_down/`basename $base`.ptb_dictionary.oracle
done
