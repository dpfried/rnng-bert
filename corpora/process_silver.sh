#!/bin/bash

train_file=silver_train.stripped
wsj_train_file=train.stripped
dev_file=dev.stripped
test_file=test.stripped
dictionary_file=silver_train.dictionary

# echo "building dictionary"
# python ../get_dictionary.py $train_file > $dictionary_file

# echo "silver dev, discrim"
# python ../get_oracle.py $dictionary_file $dev_file > silver_dev.oracle
# echo "silver test, discrim"
# python ../get_oracle.py $dictionary_file $test_file > silver_test.oracle
# echo "silver train, discrim"
# python ../get_oracle.py $dictionary_file $train_file > silver_train.oracle
echo "silver gold, discrim"
# python ../get_oracle.py $dictionary_file $wsj_train_file > silver_wsj-train.oracle
# echo "silver dev, gen"
# python ../get_oracle_gen.py $dictionary_file $dev_file > silver_dev_gen.oracle
# echo "silver test, gen"
# python ../get_oracle_gen.py $dictionary_file $test_file > silver_test_gen.oracle
echo "silver train, gen"
python ../get_oracle_gen.py $dictionary_file $wsj_train_file > silver_wsj-train_gen.oracle
