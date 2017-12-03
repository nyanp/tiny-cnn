#!/bin/bash -ex

build_dir=${BUILD_DIR:='build'}

mkdir -p $build_dir && cd $build_dir

if [[ $RUN_LINTS == ON ]] ; then
  target="test_lints"

elif [[ $BUILD_TESTS == ON ]] ; then
  cmake_args="-DBUILD_TESTS=ON"

elif [[ $BUILD_EXAMPLES == ON ]] ; then
  cmake_args="-DBUILD_EXAMPLES=ON"

fi

cmake ${cmake_args} ..

make -k -j$(nproc --ignore=1) ${target}

if [[ $BUILD_TESTS == ON ]] ; then
  ./test/tiny_dnn_test
fi

