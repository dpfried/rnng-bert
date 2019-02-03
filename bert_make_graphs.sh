#!/bin/bash

# python scripts/bert_make_graph.py --bert_model_dir bert_models/uncased_L-12_H-768_A-12/ --feature_downscale 4.0
# python scripts/bert_make_graph.py --bert_model_dir bert_models/uncased_L-24_H-1024_A-16/ --feature_downscale 6.0
python scripts/bert_make_graph.py --bert_model_dir bert_models/uncased_L-12_H-768_A-12/
python scripts/bert_make_graph.py --bert_model_dir bert_models/uncased_L-24_H-1024_A-16/
