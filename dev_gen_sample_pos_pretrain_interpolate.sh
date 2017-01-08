#!/bin/bash

out_dir="expts_sampling_interpolate"
mkdir $out_dir

num_samples=$1
discrim_model="../rnng_adhi/expts/ntparse_pos_0_2_32_128_16_128-pid28571.params_91.94"
gen_model="ntparse_gen_D0.3_2_256_256_16_256-pid4187.params"
num_sentences=1700 # dev

samples=${out_dir}/dev_pos_embeddings_s=${num_samples}.samples

trees=${samples}.trees
likelihoods=${samples}.likelihoods
rescored=${samples}.rescored.trees
rescored_preterm=${samples}.rescored.preterm.trees
log_likelihoods=${samples}.llh.txt
hyp=${samples}.hyp.trees
hyp_final=${samples}.hyp_final.trees
parsing_result=${samples}.parsing_result.txt

if [ -z "$2" ] 
then
    python utils/rescore.py $num_sentences $num_samples $samples $likelihoods > $rescored
else
    python utils/rescore.py $num_sentences $num_samples $samples $likelihoods --gen_lambda $2 > $rescored
fi

utils/add-fake-preterms-for-eval.pl $rescored > $rescored_preterm
utils/replace-unks-in-trees.pl corpora/dev.oracle $rescored_preterm > $hyp
python utils/remove_dev_unk.py corpora/dev.stripped $hyp > $hyp_final
EVALB/evalb -p EVALB/COLLINS.prm corpora/dev.stripped $hyp_final | tee $parsing_result
