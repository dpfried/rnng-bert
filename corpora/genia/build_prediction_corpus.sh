#!/bin/bash

SCRIPT_DIR="../../scripts"

STRIP_TOP=${SCRIPT_DIR}/strip_top.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py

BERT_BASE_PATH="../../bert_models/uncased_L-12_H-768_A-12/"
BERT_LARGE_PATH="../../bert_models/uncased_L-24_H-1024_A-16/"

# get cross and huang cleaned version of PTB, which strips functional annotations

DICTIONARY="train.dictionary"

mkdir predictions/top_down
mkdir predictions/in_order

for fname in predictions/*.txt
do
  bname=`basename $fname`
  bname="${bname%.*}"
  python $GET_ORACLE $DICTIONARY $fname --bert_model_dir $BERT_BASE_PATH --collapse_unary > predictions/top_down/${bname}.collapse-unary.oracle
  python $GET_ORACLE $DICTIONARY $fname --bert_model_dir $BERT_BASE_PATH --collapse_unary --in_order > predictions/in_order/${bname}.collapse-unary.oracle
done
