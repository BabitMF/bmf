#!/bin/bash


cd tests/data && ./gen.sh && cd -
export HMP_TEST_DATA_ROOT=`pwd`/tests/data
cd build && ctest