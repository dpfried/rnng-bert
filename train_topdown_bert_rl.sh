#!/bin/bash
dynet_seed=1

batch_size=16
bert_lr="2e-5"
method=reinforce
candidates=10

name="topdown_bert_bs=${batch_size}_lr=${bert_lr}_reinforce_c=${candidates}"

mkdir logs 2>&1

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 5000,5000,1000 \
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
    --min_risk_training \
    --min_risk_method $method \
    --min_risk_candidates $candidates \
    --min_risk_include_gold \
    --model_output_dir models/${name} \
    $@ \
    2>&1 | tee logs/${name}.out
