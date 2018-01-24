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

hidden_dim=$lstm_input_dim
#out_dir="expts_jan-18/"
out_dir="restart_test/"
mkdir $out_dir 2> /dev/null
export MKL_NUM_THREADS=4


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

models=($(ls -t ${output_prefix}*periodic*))
num_models=${#models[@]}
if [ "$num_models" -ne 1 ]
then
    echo "need exactly one matching model"
    echo $models
    exit 1
fi
model_to_load=${models[0]}
echo "loading model $model_to_load"

echo "saving to $output_prefix"

build/nt-parser/nt-parser \
    --cnn-seed $dynet_seed \
    --cnn-mem 2000,2000,1000 \
    -x \
    -T corpora/train.oracle \
    -d corpora/dev.oracle \
    -C corpora/dev.stripped \
    -t \
    -P \
    --pretrained_dim 100 \
    -w embeddings/sskip.100.filtered.vectors \
    --lstm_input_dim $dim \
    --hidden_dim $dim \
    -D 0.2 \
    --model $model_to_load \
    --min_risk_training \
    --min_risk_method $method \
    --min_risk_candidates $candidates \
    --min_risk_include_gold \
    --optimizer $optimizer \
    --batch_size $batch_size \
    --model_output_file $output_prefix \
    >> ${output_prefix}.stdout \
    2>> ${output_prefix}.stderr
