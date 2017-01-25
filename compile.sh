#!/bin/bash
rm -r build
mkdir build
cd build
cmake -DEIGEN3_INCLUDE_DIR="/home/ubuntu/lib/eigen" -DMKL=TRUE -DMKL_ROOT=/opt/intel/mkl ..
make -j16
