#!/bin/bash
#SBATCH --job-name=mle
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
lstm_input_dim=$2

hidden_dim=$lstm_input_dim
out_dir="expts_jan-18/"
mkdir $out_dir

export MKL_NUM_THREADS=4

output_prefix=${out_dir}/discrim_wsj_embeddings_${dynet_seed}_lstm_input_dim=${lstm_input_dim}_hidden_dim=${hidden_dim}

echo $output_prefix

# build/nt-parser/nt-parser \
#     --cnn-seed $dynet_seed \
#     --cnn-mem 2000,2000,1000 \
#     -x \
#     -T corpora/train.oracle \
#     -d corpora/dev.oracle \
#     -C corpora/dev.stripped \
#     -t \
#     -P \
#     --pretrained_dim 100 \
#     -w embeddings/sskip.100.filtered.vectors \
#     --lstm_input_dim $lstm_input_dim \
#     --hidden_dim $hidden_dim \
#     -D 0.2 \
#     --model_output_file $output_prefix \
#     > ${output_prefix}.stdout \
#     2> ${output_prefix}.stderr
