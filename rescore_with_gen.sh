#!/bin/bash

samples=$1
trees=${1}.trees
gen_model=expts/ntparse_gen_D0.3_2_256_256_16_256-pid4187.params_20.3379
likelihoods=${1}.likelihoods
rescored=${1}.rescored.trees
rescored_preterm=${1}.rescored.preterm.trees
log_likelihoods=${1}.llh.txt
hyp=${1}.hyp.trees
hyp_final=${1}.hyp_final.trees
parsing_result=${1}.parsing_result.txt

utils/cut-corpus.pl 3 $1 > $trees
build/nt-parser/nt-parser-gen -x -T corpora/train_gen.oracle --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -p $trees -m $gen_model > $likelihoods
utils/is-estimate-marginal-llh.pl 2416 100 $samples $likelihoods > $log_likelihoods 2> $rescored

utils/add-fake-preterms-for-eval.pl $rescored > $rescored_preterm
utils/replace-unks-in-trees.pl corpora/test.oracle $rescored_preterm > $hyp
utils/remove_dev_unk.py corpora/test.stripped $hyp > $hyp_final
EVALB/evalb -p COLLINS.prm corpora/test.stripped $hyp_final > $parsing_result
