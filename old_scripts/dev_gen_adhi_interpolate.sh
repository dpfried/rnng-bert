#!/bin/bash

out_dir="expts_adhi_rescore"
mkdir $out_dir

gen_model="ntparse_gen_D0.3_2_256_256_16_256-pid4187.params"
num_sentences=2416 # test
num_samples=100

samples="${out_dir}/test-samples.props"

trees=${samples}.trees
likelihoods=${samples}.likelihoods
rescored=${samples}.rescored.trees
rescored_preterm=${samples}.rescored.preterm.trees
log_likelihoods=${samples}.llh.txt
hyp=${samples}.hyp.trees
hyp_final=${samples}.hyp_final.trees
parsing_result=${samples}.parsing_result.txt

utils/cut-corpus.pl 3 $samples > $trees
build/nt-parser/nt-parser-gen -x -T corpora/train_gen.oracle --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -p $trees -m $gen_model > $likelihoods #2> /dev/null
utils/is-estimate-marginal-llh.pl $num_sentences $num_samples $samples $likelihoods > $log_likelihoods 2> $rescored
if [ -z "$2" ] 
then
    python utils/rescore.py $num_sentences $num_samples $samples $likelihoods > $rescored
else
    python utils/rescore.py $num_sentences $num_samples $samples $likelihoods --gen_lambda $2 > $rescored
fi

utils/add-fake-preterms-for-eval.pl $rescored > $rescored_preterm
utils/replace-unks-in-trees.pl corpora/test.oracle $rescored_preterm > $hyp
python utils/remove_dev_unk.py corpora/test.stripped $hyp > $hyp_final
EVALB/evalb -p EVALB/COLLINS.prm corpora/test.stripped $hyp_final | tee $parsing_result
