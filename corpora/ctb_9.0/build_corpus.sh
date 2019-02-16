#!/bin/bash

TRAIN="train.gold.original"
DEV="dev.gold.original"
TEST="test.gold.original"
OTHERS="others.gold.original"

SCRIPT_DIR="../../scripts"

REMOVE_TRACES=${SCRIPT_DIR}/remove_traces.py
FILTER_BAD=${SCRIPT_DIR}/filter_bad_trees.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py
NORMALIZE_CHINESE=${SCRIPT_DIR}/normalize_chinese_punct.py
NORMALIZE_UNICODE=${SCRIPT_DIR}/normalize_unicode.py
REPLACE_SYMBOLS=${SCRIPT_DIR}/replace_symbols.py

BERT_PATH="../../bert_models/chinese_L-12_H-768_A-12/"

python $REMOVE_TRACES < $TRAIN | python $NORMALIZE_CHINESE | python $NORMALIZE_UNICODE > train.gold.normed.stripped
python $REMOVE_TRACES < $DEV | python $NORMALIZE_CHINESE | python $NORMALIZE_UNICODE > dev.gold.normed.stripped
python $REMOVE_TRACES < $TEST | python $NORMALIZE_CHINESE | python $NORMALIZE_UNICODE > test.gold.normed.stripped
python $FILTER_BAD < $OTHERS | python $REMOVE_TRACES | python $NORMALIZE_CHINESE  | python $NORMALIZE_UNICODE > others.gold.normed.stripped.all-tags
#python $FILTER_BAD < $OTHERS | python $REMOVE_TRACES | python $NORMALIZE_CHINESE > others.gold.normed.stripped.all-tags
# important! don't evaluate against others.gold.stripped, just use it for oracle
python $REPLACE_SYMBOLS --map_from DFL EMO FLR IMG INC META OTH SKIP TYPO WHPP --map_to FRAG < others.gold.normed.stripped.all-tags > others.gold.normed.stripped

DICTIONARY="train.normed.dictionary"

python $GET_DICTIONARY train.gold.normed.stripped > $DICTIONARY

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

# discriminative oracles
for SPLIT in train.gold.normed dev.gold.normed test.gold.normed others.gold.normed
do
  echo "split: " $SPLIT
  # in_order discriminative
  python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH > in_order/${SPLIT}.oracle
  python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH --collapse_unary > in_order/${SPLIT}.collapse-unary.oracle
  python $GET_ORACLE --in_order $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH --reverse_trees > in_order/${SPLIT}.reverse-trees.oracle

  # top_down discriminative
  python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH > top_down/${SPLIT}.oracle
  python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH --collapse_unary > top_down/${SPLIT}.collapse-unary.oracle
  python $GET_ORACLE $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking --bert_model_dir $BERT_PATH --reverse_trees > top_down/${SPLIT}.reverse-trees.oracle

  # top_down generative
  python $GET_ORACLE_GEN $DICTIONARY ${SPLIT}.stripped --no_morph_aware_unking > top_down/${SPLIT}_gen.oracle
done
