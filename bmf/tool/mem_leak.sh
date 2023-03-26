#!/bin/bash

# BMF memory leak script

RUNNING_DIR=""
CASE=""
CASE_NAME=""

# Handle options
if [ $# > 0 ]
then
  RUNNING_DIR="$1"
  LOG_DIR="$PWD/$2"
  CASE="$3"
  CASE_NAME="$4"
fi

cd ${RUNNING_DIR}

COMMAND="valgrind --log-file=${LOG_DIR}/${CASE_NAME}.log --tool=memcheck --leak-check=yes --show-reachable=yes
 --show-leak-kinds=all --track-origins=yes python3 -m unittest ${CASE}"
echo ${COMMAND}

${COMMAND}

DEFINITELY_LOST_NUM=`grep -o 'definitely lost in' ${LOG_DIR}/${CASE_NAME}.log | wc -l`
ADDITION_INFO="number of definitely_lost occurrences is : ${DEFINITELY_LOST_NUM}"
echo ${ADDITION_INFO} >> ${LOG_DIR}/${CASE_NAME}.log