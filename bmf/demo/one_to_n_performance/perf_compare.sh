#!/bin/bash
num=1
if [ ! -n "$1" ]; then
    num=1
else
    num=$1
fi
echo "loop number: $num"

for i in $(seq 1 $num)
do
    python3 one_to_n_transcode.py 2>&1 | tee bmf.log
    cat bmf.log |grep "BMF time cost" > commpare_results.txt

    python3 ./runffmpegbygraph.py 2>&1 | tee ffmpeg.log
    cat ffmpeg.log |grep "FFmpeg time cost" >> commpare_results.txt
done

wait

