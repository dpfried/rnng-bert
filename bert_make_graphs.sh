#!/bin/bash

python scripts/bert_make_graph.py --bert_model_dir bert_models/uncased_L-12_H-768_A-12/
python scripts/bert_make_graph.py --bert_model_dir bert_models/uncased_L-24_H-1024_A-16/

DIR="bert_models/uncased_L-24_H-1024_A-16_selfatt-2"

python scripts/self_attention_make_graph.py \
  --num_nonbert_layers 2 \
  --bert_model_dir bert_models/uncased_L-24_H-1024_A-16/ \
  --output_checkpoint ${DIR}/bert_model.ckpt \
  --bert_output_file ${DIR}_graph.pb

# python scripts/self_attention_make_graph.py \
#   --num_nonbert_layers 2 \
#   --bert_model_dir bert_models/uncased_L-24_H-1024_A-16/ \
#   --output_checkpoint /tmp/bert_model.ckpt \
#   --extra_grad_clip 1.0 \
#   --bert_output_file ${DIR}_egc-1.0_graph.pb

python scripts/self_attention_make_graph.py \
  --disable_bert \
  --num_nonbert_layers 8 \
  --bert_output_file bert_models/nonbert_8layer_graph.pb \
  --output_checkpoint bert_models/nonbert_8layer/bert_model.ckpt \
  --nonbert_vocabulary_size 54263 # 54261 terminal vocab size, including train, + 2 for CLS and SEP

python scripts/bert_make_graph.py --bert_model_dir bert_models/chinese_L-12_H-768_A-12/
