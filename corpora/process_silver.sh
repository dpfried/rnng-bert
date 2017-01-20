#!/bin/bash

echo "silver dev, discrim"
python ../get_oracle.py silver_train.stripped dev.stripped > silver_dev.oracle
echo "silver test, discrim"
python ../get_oracle.py silver_train.stripped test.stripped > silver_test.oracle
echo "silver train, discrim"
python ../get_oracle.py silver_train.stripped silver_train.stripped > silver_train.oracle
# echo "silver dev, gen"
# python ../get_oracle_gen.py silver_train.stripped dev.stripped > silver_dev_gen.oracle
# echo "silver test, gen"
# python ../get_oracle_gen.py silver_train.stripped test.stripped > silver_test_gen.oracle
echo "silver train, gen"
python ../get_oracle_gen.py silver_train.stripped silver_train.stripped > silver_train_gen.oracle
