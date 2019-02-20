#!/bin/bash

source ../analysis_path.sh


SCRIPT_DIR="../../scripts"

STRIP_ROOT=${SCRIPT_DIR}/strip_root.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py
NORMALIZE_UNICODE=${SCRIPT_DIR}/normalize_unicode.py

BERT_PATH="../../bert_models/chinese_L-12_H-768_A-12/"

DICTIONARY="train.dictionary"
python $GET_DICTIONARY train.gold.stripped > $DICTIONARY
python $GET_DICTIONARY train.pred.stripped | diff $DICTIONARY -

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

for SPLIT in train dev test
do
  for TAGS in gold pred
  do
    STRIPPED=${SPLIT}.${TAGS}.stripped
    rm $STRIPPED
    ln -s $ANALYSIS_DIR/corpora/ctb_5.1/$STRIPPED $STRIPPED

    PROCESSED=${SPLIT}.${TAGS}.processed

    python $STRIP_ROOT --symbol TOP < $STRIPPED > $PROCESSED

    # top_down discriminative
    python $GET_ORACLE $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH > top_down/${SPLIT}.${TAGS}.oracle
    #python $GET_ORACLE $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH --collapse_unary > top_down/${SPLIT}.${TAGS}.collapse-unary.oracle
    #python $GET_ORACLE $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH --reverse_trees > top_down/${SPLIT}.${TAGS}.reverse-trees.oracle

    # top_down generative
    python $GET_ORACLE_GEN $DICTIONARY $PROCESSED --no_morph_aware_unking > top_down/${SPLIT}.${TAGS}_gen.oracle

    # in_order discriminative
    python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH > in_order/${SPLIT}.${TAGS}.oracle
    #python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH --collapse_unary > in_order/${SPLIT}.${TAGS}.collapse-unary.oracle
    #python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH --reverse_trees > in_order/${SPLIT}.${TAGS}.reverse-trees.oracle
  done
done
