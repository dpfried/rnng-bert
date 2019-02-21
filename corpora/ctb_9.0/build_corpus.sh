#!/bin/bash

source ../analysis_path.sh


SCRIPT_DIR="../../scripts"

REMOVE_TRACES=${SCRIPT_DIR}/remove_traces.py
FILTER_BAD=${SCRIPT_DIR}/filter_bad_trees.py
GET_DICTIONARY=${SCRIPT_DIR}/get_dictionary.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py
NORMALIZE_CHINESE=${SCRIPT_DIR}/normalize_chinese_punct.py
NORMALIZE_UNICODE=${SCRIPT_DIR}/normalize_unicode.py
REPLACE_SYMBOLS=${SCRIPT_DIR}/replace_symbols.py
STRIP_ROOT=${SCRIPT_DIR}/strip_root.py

BERT_PATH="../../bert_models/chinese_L-12_H-768_A-12/"

for SPLIT in train dev test
do
  STRIPPED=${SPLIT}.pred.stripped
  ln -s $ANALYSIS_DIR/corpora/ctb_9.0/$STRIPPED $STRIPPED
  PROCESSED=${SPLIT}.pred.processed

  python $STRIP_ROOT --symbol "TOP" --must_have < $STRIPPED > $PROCESSED
done

DICTIONARY="train.out-domain.dictionary"
rm $DICTIONARY
ln -s ../ctb_5.1/train.dictionary $DICTIONARY

mkdir top_down 2> /dev/null
mkdir in_order 2> /dev/null

for SPLIT in newswire broadcast_news broadcast_conversations weblogs discussion_forums chat_messages conversational_speech
do
  echo "split: " $SPLIT
  STRIPPED=${SPLIT}.pred.stripped
  ln -s $ANALYSIS_DIR/corpora/ctb_9.0/$STRIPPED $STRIPPED

  PROCESSED=${SPLIT}.pred.processed
  python $REPLACE_SYMBOLS --map_from DFL EMO FLR IMG INC META OTH SKIP TYPO WHPP --map_to FRAG < $STRIPPED \
    | python $STRIP_ROOT --symbol "TOP" --must_have > $PROCESSED

  # in_order discriminative
  python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH > in_order/${SPLIT}.pred.oracle
  # python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH --collapse_unary > in_order/${SPLIT}.pred.collapse-unary.oracle
  # python $GET_ORACLE --in_order $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH --reverse_trees > in_order/${SPLIT}.pred.reverse-trees.oracle

  # top_down discriminative
  python $GET_ORACLE $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH > top_down/${SPLIT}.pred.oracle
  # python $GET_ORACLE $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH --collapse_unary > top_down/${SPLIT}.pred.collapse-unary.oracle
  # python $GET_ORACLE $DICTIONARY $PROCESSED --no_morph_aware_unking --bert_model_dir $BERT_PATH --reverse_trees > top_down/${SPLIT}.pred.reverse-trees.oracle

  # top_down generative
  python $GET_ORACLE_GEN $DICTIONARY $PROCESSED --no_morph_aware_unking > top_down/${SPLIT}_gen.pred.oracle
done
