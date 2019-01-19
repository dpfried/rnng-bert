#!/bin/bash
#SBATCH --job-name=reinforce
#SBATCH --account=fc_bnlp
#SBATCH --partition=savio2_htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#XXSBATCH --gres=gpu:1
#SBATCH --time=71:59:00
#SBATCH --mail-user=dfried@berkeley.edu
#SBATCH --mail-type=all

source activate.sh
dynet_seed=$1
method=reinforce
candidates=$2
dim=$3
optimizer=sgd
batch_size=1

out_dir="expts_ctb/sequence_level"
mkdir -p $out_dir 2> /dev/null

output_prefix="${out_dir}/${dynet_seed}_method=${method}_candidates=${candidates}_opt=${optimizer}_include-gold"

# if [ -z "$batch_size" ]
# then
#   batch_size=1
# else
#   output_prefix="${output_prefix}_batch-size=${batch_size}"
# fi

if [ -z "$dim" ]
then
  dim=128
else
  output_prefix="${output_prefix}_dim=${dim}"
fi

echo $output_prefix

export MKL_NUM_THREADS=4

# build/nt-parser/nt-parser \
#     --cnn-seed $dynet_seed \
#     --cnn-mem 2000,2000,1000 \
#     -x \
#     -T ctb_corpora/tx-pos.oracle \
#     -d ctb_corpora/dx-pos.oracle \
#     -C ctb_corpora/dx-pos.stripped \
#     -t \
#     -P \
#     --pretrained_dim 80 \
#     -w embeddings/zzgiga.sskip.80.filtered.vectors \
#     --lstm_input_dim $dim \
#     --hidden_dim $dim \
#     -D 0.2 \
#     --min_risk_training \
#     --min_risk_method $method \
#     --min_risk_candidates $candidates \
#     --min_risk_include_gold \
#     --model_output_file $output_prefix \
#     --optimizer $optimizer \
#     --batch_size $batch_size \
#     > ${output_prefix}.stdout \
#     2> ${output_prefix}.stderr
