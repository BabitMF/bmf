#!/bin/bash

if [ $# -lt 1 ]
then
    printf "Usage: %s <files_dir>\n" $0
    exit 1
fi

export FILES_DIR=$1
export FFMPEG_LOG_OPTS="-hide_banner -loglevel warning"
#export FFMPEG_LOG_OPTS="-hide_banner"

cd colors && ./gen.sh && cd -
cd common && ./gen.sh && cd -
cd videos && ./gen.sh && cd -
