#!/bin/bash

# epoch 121.761, perplexity = 106.772
#model_file=ntparse_gen_D0.3_2_256_256_16_256-seed3-pid1475.params.bin

# epoch 104, perplexity = 108.05
model_file=ntparse_gen_D0.3_2_256_256_16_256-seed5-pid1484.params.bin

if [ "$#" -ne 5 ]; then
    echo "args: beam_size beam_size_at_word shift_size action_sig_length block_num"
    exit 1
fi

base_dir=/tmp/expts

beam_size=$1
at_word=$2
shift_size=$3
action_sig_length=$4
block_num=$5
block_count=100

prefix="rnng_gen-beam_size=${beam_size}-at_word=${at_word}-shift_size=${shift_size}-action_sig_length=${action_sig_length}"

decode_output=${base_dir}/${prefix}.decode # don't need block num, the code will append it
decode_beam=${base_dir}/${prefix}.beam # don't need block num, the code will append it
stdout=${base_dir}/${prefix}.stdout_block-${block_num}
stderr=${base_dir}/${prefix}.stderr_block-${block_num}

build/nt-parser/nt-parser-gen \
    --cnn-mem 4000,0,500 \
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
    --decode_word_action_sig_length $action_sig_length \
    --decode_shift_size $shift_size \
    --dev_output_file $decode_output \
    --dev_beam_file $decode_beam \
    --block_count $block_count \
    --block_num $block_num \
    > $stdout \
    2> $stderr
