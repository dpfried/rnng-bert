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
out_dir="expts_ctb/expts_jan-18/"
mkdir -p $out_dir 2> /dev/null

export MKL_NUM_THREADS=4

output_prefix=${out_dir}/discrim_embeddings_${dynet_seed}_lstm_input_dim=${lstm_input_dim}_hidden_dim=${hidden_dim}

echo $output_prefix

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 1000,1000,800 \
    -x \
    -T ctb_corpora/tx-pos.oracle \
    -d ctb_corpora/dx-pos.oracle \
    -C ctb_corpora/dx-pos.stripped \
    -t \
    -P \
    --pretrained_dim 80 \
    -w embeddings/zzgiga.sskip.80.filtered.vectors \
    --lstm_input_dim $lstm_input_dim \
    --hidden_dim $hidden_dim \
    -D 0.2 \
    --model_output_file $output_prefix \
    > ${output_prefix}.stdout \
    2> ${output_prefix}.stderr
