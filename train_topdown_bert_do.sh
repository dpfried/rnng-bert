#!/bin/bash
dynet_seed=1

batch_size=16
bert_lr="2e-5"

candidates=2
method="sample"
dpe=1.0

name="topdown_bert_bs=${batch_size}_lr=${bert_lr}_do_c=${candidates}_method=${method}_dpe=${dpe}_include-gold"

mkdir logs 2>&1

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 3000,3000,500 \
    -T corpora/english/top_down/train.oracle \
    -d corpora/english/top_down/dev.oracle \
    -C corpora/english/dev.stripped \
    -t \
    --bert \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --batch_size $batch_size \
    --bert_lr $bert_lr \
    --dynamic_exploration_candidates $candidates \
    --dynamic_exploration $method \
    --dynamic_exploration_probability $dpe \
    --dynamic_exploration_include_gold \
    --model_output_dir models/${name} \
    $@ \
    2>&1 | tee logs/${name}.out
