#!/bin/bash

# get cross and huang cleaned version of ptb, which strips functional annotations

TRAIN="02-21.10way.clean"
DEV="22.auto.clean"
TEST="23.auto.clean"

DATA_URL="https://raw.githubusercontent.com/jhcross/span-parser/d6ca8e3f2a5b7eda8d06dce85663456f8a92efbc/data/"

SCRIPT_DIR="../../scripts"

STRIP_TOP=${SCRIPT_DIR}/strip_top.py
GET_ORACLE=${SCRIPT_DIR}/get_oracle.py
GET_ORACLE_GEN=${SCRIPT_DIR}/get_oracle_gen.py

wget ${DATA_URL}${TRAIN}
wget ${DATA_URL}${DEV}
wget ${DATA_URL}${TEST}

python ${STRIP_TOP} < $TRAIN > train.stripped
python ${STRIP_TOP} < $DEV > dev.stripped
python ${STRIP_TOP} < $TEST > test.stripped

mkdir top_down

# discriminative oracles
python ${GET_ORACLE} train.stripped train.stripped > top_down/train.oracle
python ${GET_ORACLE} train.stripped dev.stripped > top_down/dev.oracle
python ${GET_ORACLE} train.stripped test.stripped > top_down/test.oracle

# generative oracles
python ${GET_ORACLE_GEN} train.stripped train.stripped > top_down/train_gen.oracle
python ${GET_ORACLE_GEN} train.stripped dev.stripped > top_down/dev_gen.oracle
python ${GET_ORACLE_GEN} train.stripped test.stripped > top_down/test_gen.oracle

mkdir in_order

# discriminative oracles
python ${GET_ORACLE} --in_order train.stripped train.stripped > in_order/train.oracle
python ${GET_ORACLE} --in_order train.stripped dev.stripped > in_order/dev.oracle
python ${GET_ORACLE} --in_order train.stripped test.stripped > in_order/test.oracle
