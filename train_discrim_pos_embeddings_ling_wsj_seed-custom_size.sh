#!/bin/bash
source activate.sh
dynet_seed=$1
lstm_input_dim=$2
hidden_dim=$lstm_input_dim
out_dir="expts_jan-18/"
mkdir $out_dir

output_prefix=${out_dir}/discrim_wsj_embeddings_${dynet_seed}_lstm_input_dim=${lstm_input_dim}_hidden_dim=${hidden_dim}

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 2000,2000,1000 \
    -x \
    -T corpora/train.oracle \
    -d corpora/dev.oracle \
    -C corpora/dev.stripped \
    -t \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.vectors \
    --lstm_input_dim $lstm_input_dim \
    --hidden_dim $hidden_dim \
    -D 0.2 \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
