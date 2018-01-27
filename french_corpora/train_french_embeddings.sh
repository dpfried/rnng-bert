#!/bin/bash
# https://github.com/wlin12/wang2vec
# git revision cd11b31ae18dd3c04fb6ae147d62e7cbbbf672c0

train_file=/data/dfried/data/wiki/frwiki-20180120-pages-articles-multistream.txt
output_file=/data/dfried/data/vectors/french.wiki.sskip.100.vectors

./word2vec -train $train_file \
    -output $output_file \
    -type 3 \
    -size 100 \
    -window 5 \
    -negative 10 \
    -nce 0 \
    -hs 0 \
    -sample 1e-4 \
    -threads 8 \
    -binary 1 \
    -iter 5 \
    -cap 0 \
    -min-count 40
