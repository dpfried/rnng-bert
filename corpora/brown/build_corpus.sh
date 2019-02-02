#!/bin/bash

RAW="Brown.goldtags.train"
STRIPPED=${RAW}.stripped

SCRIPT_DIR="../../scripts"

STRIP_TOP=${SCRIPT_DIR}/strip_top.py
STRIP_FUNCTIONAL=${SCRIPT_DIR}/strip_functional.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py

BERT_BASE_PATH="../../bert_models/uncased_L-12_H-768_A-12/"
BERT_LARGE_PATH="../../bert_models/uncased_L-24_H-1024_A-16/"

python ${STRIP_TOP} < $RAW | python $STRIP_FUNCTIONAL --remove_symbols ADV AUX EDITED NEG TYPO > $STRIPPED

DICTIONARY="train.dictionary"

python $GET_DICTIONARY $STRIPPED > $DICTIONARY

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null


# in_order discriminative
python $GET_ORACLE --in_order $DICTIONARY $STRIPPED --bert_model_dir $BERT_BASE_PATH > in_order/${RAW}.oracle
python $GET_ORACLE --in_order $DICTIONARY $STRIPPED --bert_model_dir $BERT_BASE_PATH --collapse_unary > in_order/${RAW}.collapse-unary.oracle

# discriminative oracles
# top_down discriminative
python $GET_ORACLE $DICTIONARY $STRIPPED --bert_model_dir $BERT_BASE_PATH > top_down/${RAW}.oracle
python $GET_ORACLE $DICTIONARY $STRIPPED --bert_model_dir $BERT_BASE_PATH --collapse_unary > top_down/${RAW}.collapse-unary.oracle

# top_down generative
python $GET_ORACLE_GEN $DICTIONARY $STRIPPED > top_down/${RAW}_gen.oracle
