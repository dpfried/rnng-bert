#!/bin/bash
source activate.sh
BOOST_DIR="${HOME}/lib/boost_1_58_0/"
EIGEN_DIR="${HOME}/projects/eigen_dev/"
MKL_DIR="/global/software/sl-7.x86_64/modules/langs/intel/2016.4.072/mkl"
CC_EXEC="/global/software/sl-7.x86_64/modules/langs/gcc/6.3.0/bin/gcc"
CXX_EXEC="/global/software/sl-7.x86_64/modules/langs/gcc/6.3.0/bin/g++"
mkdir build_cuda
cd build_cuda
cmake \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_DIR \
    -DMKL=TRUE \
    -DMKL_ROOT=$MKL_DIR \
    -DCMAKE_CXX_COMPILER=$CXX_EXEC \
    -DCMAKE_C_COMPILER=$CC_EXEC \
    -DBOOST_ROOT=$BOOST_DIR \
    -DBoost_NO_SYSTEM_PATHS=TRUE \
    -DBoost_NO_BOOST_CMAKE=TRUE \
    -DBOOST_DIR=$BOOST_DIR \
    -DBoost_INCLUDE_DIR=$BOOST_DIR \
    -DBACKEND=cuda \
    ..
make -j16
