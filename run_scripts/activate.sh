#!/bin/bash
export LD_LIBRARY_PATH="${HOME}/lib/boost_1_58_0/stage/lib/:$LD_LIBRARY_PATH"
export MODULEPATH=/global/software/sl-7.x86_64/modfiles/tools:/global/software/sl-7.x86_64/modfiles/langs
export MODULEPATH=$MODULEPATH:/global/home/groups/fc_bnlp/software/modfiles

module load gcc/6.3.0
module load mkl/2016.4.072
module load cuda/9.0
