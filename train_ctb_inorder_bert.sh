#!/bin/bash

dynet_seed=$1

if [ -z $dynet_seed ]; then
        dynet_seed=1
else
        shift
fi

batch_size=32
bert_lr="2e-5"

lr_decay_patience=2

prefix="inorder_bert_bs=${batch_size}_lr=${bert_lr}_adam_patience=${lr_decay_patience}_seed=${dynet_seed}"

mkdir models_ctb 2> /dev/null 
mkdir logs_ctb 2> /dev/null

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 3000,3000,500 \
    --git_state \
    -T corpora/ctb_5.1/in_order/train.gold.oracle \
    -d corpora/ctb_5.1/in_order/dev.gold.oracle \
    -C corpora/ctb_5.1/dev.stripped \
    --inorder \
    -t \
    --bert \
    --bert_model_dir bert_models/chinese_L-12_H-768_A-12 \
    --bert_graph_path bert_models/chinese_L-12_H-768_A-12_graph.pb \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --batch_size $batch_size \
    --subbatch_max_tokens 500 \
    --eval_batch_size 8 \
    --bert_lr $bert_lr \
    --lr_decay_patience $lr_decay_patience \
    --model_output_dir models_ctb/${prefix} \
    --optimizer adam \
    $@ \
    2>&1 | tee logs_ctb/${prefix}.out
