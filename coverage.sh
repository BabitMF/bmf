#!/bin/bash

# script for generating coverage report
#
# To generate coverage report for whole project(default):
#   ./coverage.sh
#
# To generate coverage report for specified dir:
#   ${dir_name} includes: sdk engine module
#   ./coverage.sh ${dir_name}
#
# To clean up coverage report:
#   ./coverage.sh clean
#

echo "begin generating coverage report"
lcov -d build -z

# Handle options
if [ $@ > 1 ]
then
    # Clean up
    if [ $1 = "clean" ]
    then
        rm -rf coverage_report
        rm -rf coverage_info
        exit
    elif [ $1 = "all" ]
    then
        cd build/output/bmf/bin
        ./test_bmf_module_sdk
        ./test_builtin_modules
        ./test_bmf_engine
        cd -
        lcov -d build -b . --no-external -c -o coverage_info
    elif [ $1 = "sdk" ]
    then
        cd build/output/bmf/bin
        ./test_bmf_module_sdk
        cd -
        lcov -d build -b . --no-external -c -o coverage_info_tmp
        lcov -e coverage_info_tmp "*sdk*" -o coverage_info
        rm coverage_info_tmp
    elif [ $1 = "module" ]
    then
        cd build/output/bmf/bin
        ./test_builtin_modules
        cd -
        lcov -d build -b . --no-external -c -o coverage_info_tmp
        lcov -e coverage_info_tmp "*c_modules*" -o coverage_info
        rm coverage_info_tmp
    elif [ $1 = "engine" ]
    then
        cd build/output/bmf/bin
        ./test_bmf_engine
        cd -
        lcov -d build -b . --no-external -c -o coverage_info_tmp
        lcov -e coverage_info_tmp "*engine*" -o coverage_info
        rm coverage_info_tmp
    else
        echo "incorrect coverage option"
        exit
    fi
else
    cd build/output/bmf/bin
    ./test_bmf_module_sdk
    ./test_builtin_modules
    ./test_bmf_engine
    cd -
    lcov -d build -b . --no-external -c -o coverage_info
fi

genhtml coverage_info -o coverage_report
echo "coverage report done"