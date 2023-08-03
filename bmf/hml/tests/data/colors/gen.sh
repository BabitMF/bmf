#!/bin/sh

JPEG_FILE=../../../../../output/bmf/files/Color-wheel-light-color-spectrum.jpeg

# pix_fmt colorspace out
gen_yuv_rgb()
{
    # convert to yuv
    ffmpeg $FFMPEG_LOG_OPTS  -y -i $JPEG_FILE -c:v rawvideo\
        -pix_fmt $1 -vf colorspace=$2:iall=bt470bg:fast=1 $3.yuv

    # gen ref rgb data
    ffmpeg $FFMPEG_LOG_OPTS -y -f rawvideo -vcodec rawvideo -s 800x800 -r 25 \
        -pix_fmt $1 -colorspace $2 -i $3.yuv -pix_fmt rgb24 $3_RGB24.yuv

    # gen ref yuv data
    ffmpeg $FFMPEG_LOG_OPTS -y -f rawvideo -vcodec rawvideo -s 800x800 -r 25 \
        -pix_fmt rgb24 -colorspace $2 -i $3_RGB24.yuv -pix_fmt $1 $3_YUV.yuv
}


#
gen_yuv_rgb yuv420p bt709 H420
gen_yuv_rgb yuv422p bt709 H422
gen_yuv_rgb yuv444p bt709 H444

#
gen_yuv_rgb yuv420p bt470bg I420
gen_yuv_rgb yuv422p bt470bg I422
gen_yuv_rgb yuv444p bt470bg I444

gen_yuv_rgb nv21 bt470bg NV21
gen_yuv_rgb nv12 bt470bg NV12

