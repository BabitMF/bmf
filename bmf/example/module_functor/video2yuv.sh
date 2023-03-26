#!/bin/sh

FFMPEG="ffmpeg  -hide_banner -loglevel error"

echo "convert to *.png"
$FFMPEG -i $1 -vsync 0 -frame_pts true out-%04d.png
for i in *.png; do
	base=$(basename "$i" ".png")
    echo "$i to $base.yuv"
    $FFMPEG -i $i -pix_fmt yuv420p $base.yuv
    rm $i
done
