#!/bin/bash

# need to first download GoogleNews-vectors-negative300.bin.gz from https://docs.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
# (linked from code.google.com/archive/p/word3vec)

# get bin to text conversion format
git clone https://github.com/marekrei/convertvec.git
cd convertvec
git checkout f54e149ee4b807ac5bea79662cae179b40984a99
make
cd ..

# convert to text format
./convertvec/convertvec bin2txt GoogleNews-vectors-negative300.bin GoogleNews-vectors-negative300.txt

# filter words in ptb
python filter_w2v.py < GoogleNews-vectors-negative300.txt > GoogleNews-ptb_filtered-vectors-negative300.txt
