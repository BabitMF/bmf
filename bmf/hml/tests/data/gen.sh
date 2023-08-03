#!/bin/bash

export FFMPEG_LOG_OPTS="-hide_banner -loglevel warning"
#export FFMPEG_LOG_OPTS="-hide_banner"

cd colors && ./gen.sh && cd -
cd common && ./gen.sh && cd -
cd videos && ./gen.sh && cd -
