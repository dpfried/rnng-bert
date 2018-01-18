#!/bin/bash
BOOST_DIR="${HOME}/lib/boost_1_58_0/"
EIGEN_DIR="${HOME}/projects/eigen/"
DYNET_ROOT="${HOME}/projects/dynet_rnng/"
MKL_DIR="/opt/intel/mkl"
mkdir build
cd build
cmake \
    -DEIGEN3_INCLUDE_DIR=$EIGEN_DIR \
    -DMKL=TRUE \
    -DMKL_ROOT=$MKL_DIR \
    -DCMAKE_CXX_COMPILER=`which g++-5` \
    -DCMAKE_C_COMPILER=`which gcc-5` \
    -DBOOST_ROOT=$BOOST_DIR \
    -DBoost_NO_SYSTEM_PATHS=TRUE \
    -DBoost_NO_BOOST_CMAKE=TRUE \
    -DBOOST_DIR=$BOOST_DIR \
    -DBoost_INCLUDE_DIR=${BOOST_DIR} \
    -DDYNET_INCLUDE_DIR=${DYNET_ROOT} \
    -DDYNET_LIBRARIES=${DYNET_ROOT}/build/dynet \
    ..
make -j16
