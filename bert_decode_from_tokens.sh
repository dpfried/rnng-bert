#!/bin/bash
sentsplit_token_file=$1
working_dir=$2

output_file="${sentsplit_token_file}.parsed"

echo "will write output file to $output_file"

dictionary_file="corpora/english/train.dictionary"
model_dir="models/uncased-english-wwm"

train_oracle_file="corpora/english/in_order/train.oracle"

if [ -z $working_dir ]
then
        working_dir=`mktemp -d`
        echo "writing temporary files in $working_dir"
fi

oracle_file="${working_dir}/`basename $sentsplit_token_file`.oracle"
sentsplit_decode_file="${working_dir}/`basename $sentsplit_token_file`.sentsplit-decodes"
long_sentences_file="${working_dir}/`basename $sentsplit_token_file`.long-sents"

python3 scripts/get_oracle.py \
        $dictionary_file \
        $sentsplit_token_file  \
        --is_token_file \
        --bert_model_dir $model_dir \
        --base_fname $oracle_file \
        --long_sentences_output_file $long_sentences_file

build/nt-parser/nt-parser \
        --cnn-seed 1 \
        --cnn-mem 3000,0,500 \
        --model_dir $model_dir \
        -T $train_oracle_file \
        --test_data $oracle_file \
        --test_data_no_trees \
        --lstm_input_dim 128 \
        --hidden_dim 128 \
        --beam_size 1 \
        --batch_size 8 \
        --text_format \
        --inorder \
        --bert \
        --bert_large \
        --ptb_output_file $sentsplit_decode_file \
        --output_trees_only \
        --output_beam_as_samples \
        > /dev/null

python3 scripts/cat_decodes.py \
        $sentsplit_token_file \
        $sentsplit_decode_file \
        $output_file \
        --bert_model_dir $model_dir \
        --long_sentences_file $long_sentences_file

echo `wc -l < $sentsplit_token_file` lines in input file, $sentsplit_token_file
echo `wc -l < $output_file` lines in output file, $output_file
