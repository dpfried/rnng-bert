#!/bin/bash
corpus=$1
block_number=$2
build/nt-parser/nt-parser \
        --cnn-seed 1 \
        --cnn-mem 3000,0,500 \
        --model_dir models/uncased-english-wwm \
        -T corpora/english/in_order/train.oracle \
        --test_data ../../corpora/oracles/${corpus}_raw_subsampled_tokenized_sentsplit.token-oracle-block-$block_number \
        --test_data_no_trees \
        --lstm_input_dim 128 \
        --hidden_dim 128 \
        --beam_size 1 \
        --batch_size 8 \
        --text_format \
        --inorder \
        --bert \
        --bert_large \
        --ptb_output_file ../../corpora/decodes/${corpus}_raw_subsampled_tokenized_sentsplit.tagged-decode-block-$block_number \
        --output_trees_only \
        --output_beam_as_samples \
        > ../../corpora/decodes/${corpus}_raw_subsampled_tokenized_sentsplit.decode-block-$block_number
