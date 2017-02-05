#!/bin/bash

# epoch 121.761, perplexity = 106.772
model_file=ntparse_gen_D0.3_2_256_256_16_256-seed3-pid1475.params.bin

if [ "$#" -ne 3 ]; then
    echo "args: beam_size beam_size_at_word block_num"
    exit 1
fi

beam_size=$1
at_word=$2
block_num=$3
block_count=100
prefix="rnng_gen-beam_size=${beam_size}-at_word=${at_word}"
decode_output=expts/${prefix}.decode # don't need block num, the code will append it
decode_beam=expts/${prefix}.beam # don't need block num, the code will append it
stdout=expts/${prefix}.stdout_block-${block_num}
stderr=expts/${prefix}.stderr_block-${block_num}

build/nt-parser/nt-parser-gen \
    --cnn-mem 45000,0,500 \
    -T corpora/train_gen.oracle \
    -d corpora/dev_gen.oracle \
    -C corpora/dev.stripped \
    --clusters clusters-train-berk.txt \
    --input_dim 256 \
    --lstm_input_dim 256 \
    --hidden_dim 256 \
    -m  $model_file \
    --greedy_decode_dev \
    --beam_within_word \
    --word_completion_is_shift \
    --decode_beam_size $beam_size \
    --decode_beam_filter_at_word_size $at_word \
    --dev_output_file $decode_output \
    --dev_beam_file $decode_beam \
    --block_count $block_count \
    --block_num $block_num \
    > $stdout \
    2> $stderr
