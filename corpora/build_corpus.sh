#!/bin/bash

# get cross and huang cleaned version of ptb, which strips functional annotations

TRAIN="02-21.10way.clean"
DEV="22.auto.clean"
TEST="23.auto.clean"

DATA_URL="https://raw.githubusercontent.com/jhcross/span-parser/d6ca8e3f2a5b7eda8d06dce85663456f8a92efbc/data/"

wget ${DATA_URL}${TRAIN}
wget ${DATA_URL}${DEV}
wget ${DATA_URL}${TEST}

python strip_top.py < $TRAIN > train.stripped
python strip_top.py < $DEV > dev.stripped
python strip_top.py < $TEST > test.stripped

# discriminative oracles
python ../get_oracle.py train.stripped train.stripped > train.oracle
python ../get_oracle.py train.stripped dev.stripped > dev.oracle
python ../get_oracle.py train.stripped test.stripped > test.oracle

# generative oracles
python ../get_oracle_gen.py train.stripped train.stripped > train_gen.oracle
python ../get_oracle_gen.py train.stripped dev.stripped > dev_gen.oracle
python ../get_oracle_gen.py train.stripped test.stripped > test_gen.oracle
