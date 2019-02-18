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

prefix="inorder_self-att_bs=${batch_size}_lr=${bert_lr}_adam_patience=${lr_decay_patience}_seed=${dynet_seed}"

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 3000,3000,500 \
    --git_state \
    -T corpora/english/in_order/train.oracle \
    -d corpora/english/in_order/dev.oracle \
    -C corpora/english/dev.stripped \
    --inorder \
    -t \
    --bert \
    --self_attend_words \
    --bert_model_dir bert_models/nonbert_8layer/ \
    --bert_graph_path bert_models/nonbert_8layer_graph.pb \
    --lstm_input_dim 128 \
    --hidden_dim 128 \
    -D 0.2 \
    --batch_size $batch_size \
    --subbatch_max_tokens 500 \
    --eval_batch_size 8 \
    --bert_lr $bert_lr \
    --lr_decay_patience $lr_decay_patience \
    --bert_large \
    --model_output_dir models/${prefix} \
    --optimizer adam \
    $@ \
    2>&1 | tee logs/${prefix}.out
