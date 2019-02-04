#!/bin/bash
model_dir=$1
dev_or_test=$2
beam_size=$3
shift
shift
shift

if [ -z $beam_size ]; then
    beam_size=1
fi

if [ -z $dev_or_test ]; then
    dev_or_test=dev
fi

batch_size=8

mkdir k_best 2> /dev/null

build/nt-parser/nt-parser \
    --cnn-seed 1 \
    --cnn-mem 5000,5000,500 \
    --model_dir $model_dir \
    -T corpora/english/top_down/train.oracle \
    -p corpora/english/top_down/${dev_or_test}.oracle \
    --bracketing_test_data corpora/english/${dev_or_test}.stripped \
    --bert \
    --bert_large \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    --beam_size $beam_size \
    --batch_size $batch_size \
    --output_beam_as_samples \
    --ptb_output_file k_best/`basename $model_dir`-${dev_or_test}-${beam_size}.txt \
    $@
