#!/bin/bash

TRAIN="02-21.10way.clean"
DEV="22.auto.clean"
TEST="23.auto.clean"

DATA_URL="https://raw.githubusercontent.com/jhcross/span-parser/d6ca8e3f2a5b7eda8d06dce85663456f8a92efbc/data/"

SCRIPT_DIR="../../scripts"

STRIP_TOP=${SCRIPT_DIR}/strip_top.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py

BERT_BASE_PATH="../../bert_models/uncased_L-12_H-768_A-12/"
BERT_LARGE_PATH="../../bert_models/uncased_L-24_H-1024_A-16/"

# get cross and huang cleaned version of PTB, which strips functional annotations

for fname in $TRAIN $DEV $TEST
do
  rm $fname 2> /dev/null
done

wget ${DATA_URL}${TRAIN}
wget ${DATA_URL}${DEV}
wget ${DATA_URL}${TEST}

python ${STRIP_TOP} < $TRAIN > train.stripped
python ${STRIP_TOP} < $DEV > dev.stripped
python ${STRIP_TOP} < $TEST > test.stripped

DICTIONARY="train.dictionary"

python $GET_DICTIONARY train.stripped > $DICTIONARY

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

# discriminative oracles
for SPLIT in train dev test
do
  # top_down discriminative
  python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --bert_model_dir $BERT_BASE_PATH > top_down/${SPLIT}.oracle
  python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --bert_model_dir $BERT_BASE_PATH --collapse_unary > top_down/${SPLIT}.collapse-unary.oracle

  # the bert_large oracle files are identical because Base and Large, for English uncased at least, have the same vocab.txt
  #python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --bert_model_dir $BERT_LARGE_PATH > top_down/${SPLIT}_bert_large.oracle

  # top_down generative
  python $GET_ORACLE_GEN $DICTIONARY ${SPLIT}.stripped > top_down/${SPLIT}_gen.oracle

  # in_order discriminative
  python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --bert_model_dir $BERT_BASE_PATH > in_order/${SPLIT}.oracle
  python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --bert_model_dir $BERT_BASE_PATH --collapse_unary > in_order/${SPLIT}.collapse-unary.oracle

  # the bert_large oracle files are identical because Base and Large, for English uncased at least, have the same vocab.txt
  #python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --bert_model_dir $BERT_LARGE_PATH > in_order/${SPLIT}_bert_large.oracle
done
