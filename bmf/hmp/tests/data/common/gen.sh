#!/bin/sh


# pix_fmt colorspace out
jpeg_to_rgb()
{
    # convert to yuv
    ffmpeg $FFMPEG_LOG_OPTS  -y -i $1 -c:v rawvideo -pix_fmt rgb24 $2_RGB24.yuv
}


#
jpeg_to_rgb ${FILES_DIR}/Lenna.png Lenna
