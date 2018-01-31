#!/bin/bash
model=$1
dim=$2
beam_size=$3
dev_or_test=$4
if [ -z $dev_or_test ]; then
    dev_or_test=dev
fi

source activate.sh

build/nt-parser/nt-parser \
    --cnn-seed 1 \
    --cnn-mem 1000,0,500 \
    --model $model \
    -x \
    -T french_corpora/train.oracle \
    -p french_corpora/${dev_or_test}.oracle \
    -C french_corpora/${dev_or_test}.stripped \
    --spmrl \
    -P \
    --lstm_input_dim $dim \
    --hidden_dim $dim \
    --beam_size $beam_size \
    -D 0.2
