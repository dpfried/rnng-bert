#!/bin/bash

out_dir="expts_beam_rescoring"
mkdir $out_dir

beam_size=$1
discrim_model="../rnng_adhi/expts/ntparse_pos_0_2_32_128_16_128-pid28571.params_91.94"
gen_model="ntparse_gen_D0.3_2_256_256_16_256-pid4187.params"
num_sentences=2416 # test

samples=${out_dir}/test_pos_embeddings_beam=${beam_size}.samples
ptb_samples=${out_dir}/test_pos_embeddings_beam=${beam_size}.ptb_samples

trees=${samples}.trees
likelihoods=${samples}.rnng-gen-likelihoods
rescored=${samples}.rescored.trees
rescored_preterm=${samples}.rescored.preterm.trees
log_likelihoods=${samples}.llh.txt
hyp=${samples}.hyp.trees
hyp_final=${samples}.hyp_final.trees
parsing_result=${samples}.parsing_result.txt

# build/nt-parser/nt-parser --cnn-mem 7000,0,800 -x -T corpora/train.oracle -p corpora/test.oracle -C corpora/test.stripped --pretrained_dim 100 -w embeddings/sskip.100.filtered.vectors -P --lstm_input_dim 128 --hidden_dim 128 -m $discrim_model --alpha 0.8 --beam_size $beam_size --output_beam_as_samples --ptb_output_file $ptb_samples > $samples 2> ${samples}.stderr
# utils/cut-corpus.pl 3 $samples > $trees
# build/nt-parser/nt-parser-gen --cnn-mem 1700 -x -T corpora/train_gen.oracle --clusters clusters-train-berk.txt --input_dim 256 --lstm_input_dim 256 --hidden_dim 256 -p $trees -m $gen_model > $likelihoods 2> /dev/null
# utils/is-estimate-marginal-llh.pl $num_sentences $beam_size $samples $likelihoods > $log_likelihoods 2> $rescored
if [ -z "$2" ] 
then
    python utils/rescore.py $num_sentences $beam_size $samples $likelihoods > $rescored
else
    python utils/rescore.py $num_sentences $beam_size $samples $likelihoods --gen_lambda $2 > $rescored
fi
utils/add-fake-preterms-for-eval.pl $rescored > $rescored_preterm
utils/replace-unks-in-trees.pl corpora/test.oracle $rescored_preterm > $hyp
python utils/remove_dev_unk.py corpora/test.stripped $hyp > $hyp_final
EVALB/evalb -p EVALB/COLLINS.prm corpora/test.stripped $hyp_final | tee $parsing_result
