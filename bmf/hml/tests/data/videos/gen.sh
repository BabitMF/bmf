#!/bin/sh

VIDEO_FILE=../../../../../output/bmf/files/big_bunny_10s_30fps.mp4
#LOG_OPTS="-hide_banner -loglevel warning"
LOG_OPTS="-hide_banner"


# pix_fmt colorspace out
cvt_pix_fmt()
{
    # convert to yuv
    if [ ! -f "$3.mp4" ]; then
        ffmpeg $FFMPEG_LOG_OPTS  -y -i $VIDEO_FILE -c:v libx264\
            -pix_fmt $1 -vf colorspace=$2:iall=bt470bg:fast=1 -c:a aac $3.mp4
    else
        echo "$3.mp4 already exists"
    fi
}

#
cvt_pix_fmt yuv420p bt709 H420
cvt_pix_fmt yuv422p bt709 H422
cvt_pix_fmt yuv444p bt709 H444

#cvt_pix_fmt nv21 bt470bg NV21
#cvt_pix_fmt nv12 bt470bg NV12
