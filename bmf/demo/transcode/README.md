## About this demo

This demo describes step-by-step how to use BMF to develop a transcoding program, including video transcoding, audio transcoding, and image transcoding. In it, you can familiarize yourself with how to use BMF and how to use FFmpeg-compatible options to achieve the capabilities you need.

This README document is a copy of colab, go to colab for a quick experience: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BabitMF/bmf/blob/master/bmf/demo/transcode/bmf_transcode_demo.ipynb)

## 1. Environmental preparation

### 1.1 FFmpeg

FFmpeg 4.x or 5.x is needed by BMF when transcoding, check versions via apt:

```bash
apt show ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavresample-dev libavutil-dev libpostproc-dev libswresample-dev libswscale-dev | grep "^Package:\|^Version:"
```

If the version meets the requirements, install ffmpeg via apt:

```bash
apt install -y ffmpeg libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev libavresample-dev libavutil-dev libpostproc-dev libswresample-dev libswscale-dev
```

Otherwise, you need to compile ffmpeg from source, you can use the script we provided(only linux and macos now):

```bash
git clone https://github.com/BabitMF/bmf bmf
cd bmf
./scripts/build_ffmpeg.sh x264 x265
```

### 1.2 BMF

BMF can be installed in many ways, we use pip here:

```bash
pip3 install BabitMF
```

## 2. Transcode demo

### 2.1 video transcoding

#### 2.1.1 remuxing demo

This demo shows how to demux a video in mp4 format and remuxing it into hls slices without involving audio and video decoding and encoding, write the following code into the file named remuxing_demo.py:

```python
import bmf

input_video_path = "../../files/big_bunny_10s_30fps.mp4"
output_path = "./remux_output.m3u8"

# create graph
graph = bmf.graph()

# decode
video = graph.decode({
    "input_path": input_video_path,
    "video_codec": "copy",
    "audio_codec": "copy"
})

(
    bmf.encode(
        video['video'],
        video['audio'],
        {
            "output_path": output_path,
            "format": "hls",
            "mux_params": {
                "hls_list_size": "0",
                "hls_time": "2",
                "hls_segment_filename": "./file%03d.ts"
            }
        }
    ).run()
)
```

Let's run it and check the result with `ffprobe`:

```bash
python3 remuxing_demo.py
ffprobe remux_output.m3u8
rm -rf remux_output.m3u8 file*.ts
```

#### 2.1.2 decoding and encoding demo

This demo shows how to decode a H.264 video, drop the audio, scale the video to 720x576p resolution, and encode it with x265 encoder. You can even do it in one line of code, write code into the file named decoding_encoding.py:

```python
import bmf

input_video_path = "./big_bunny_10s_30fps.mp4"
output_path = "./decode_scale_encode_output.mp4"
(
    bmf.graph()
        .decode({'input_path': input_video_path})['video']
        .scale(720, 576)
        .encode(None, {
            "output_path": output_path,
            "video_params": {
                "codec": "libx265"
            }
        }).run()
)
```

Run and check the result:

```bash
python3 decoding_encoding.py
ffprobe decode_scale_encode_output.mp4
rm -rf decode_scale_encode_output.mp4
```

#### 2.1.3 multi stream demo

This demo shows how to decode the original video, use the ffmpeg filter capability, and encode multi videos with different options, write code into the file named multi_stream.py:

```python
import bmf

input_video_path = "./big_bunny_10s_30fps.mp4"
output_path0 = "./decode_encode_multi_output0.mp4"
output_path1 = "./decode_encode_multi_output1.mp4"

streams = bmf.graph().decode({'input_path': input_video_path})
split_streams = streams['video'].split()
bmf.encode(split_streams[0], streams['audio'], {
    "output_path": output_path0,
    "video_params": {
        "codec": "libx264",
        "x264-params": "ssim=1:psnr=1"
    }
})
(
    bmf.encode(split_streams[1], streams['audio'], {
            "output_path": output_path1,
            "video_params": {
                "codec": "libx265",
                "preset": "fast",
                "crf": "23"
            }
        }).run()
)
```

Run and check the result:

```bash
python3 multi_stream.py

ffprobe decode_encode_multi_output0.mp4
echo "----------------------------------------------------------------------"
ffprobe decode_encode_multi_output1.mp4

rm -rf decode_encode_multi_output0.mp4 decode_encode_multi_output1.mp4
```

### 2.2 audio transcoding

This demo shows how to use BMF to decode the input video, extract the audio part, add a piece of null audio before and after, and then encode the output wav file, write code into the file named audio_transcoding.py:

```python
import bmf

input_video_path = "./big_bunny_10s_30fps.mp4"
output_path = "./with_null_audio.wav"

# create graph
graph = bmf.graph()

# decode
streams = graph.decode({
    "input_path": input_video_path
})

# create a null audio stream
audio_stream1 = graph.anullsrc('r=48000', 'cl=2').atrim('start=0', 'end=6')
audio_stream2 = graph.anullsrc('r=48000', 'cl=2').atrim('start=0', 'end=6')
concat_audio = (
    bmf.concat(audio_stream1, streams['audio'], audio_stream2, n=3, v=0, a=1)
)

(
    bmf.encode(
        None,
        concat_audio,
        {
            "output_path": output_path,
            "audio_params": {
                "codec": "aac",
                "bit_rate": 128000,
                "sample_rate": 44100,
                "channels": 2
            }
        }
    )
    .run()
)
```

Run and check the result:

```bash
python3 audio_transcoding.py
ffprobe with_null_audio.wav
rm -rf with_null_audio.wav
```

### 2.3 image transcoding

This demo shows how to decode a video, detect and skip black frames, encode and output to the pipeline, and then write files at the application layer. You can also do other custom operations, such as uploading directly to the cloud to avoid disk IO. Code is here and write it into the file named image_transcoding.py:

```python
import bmf

input_video_path = "./big_bunny_10s_30fps.mp4"

graph = bmf.graph()
streams = graph.decode({
    "input_path": input_video_path,
})
video = streams['video'].ff_filter("blackframe", threshold=32).ff_filter("metadata", "select:key=lavfi.blackframe.pblack:value=96:function=less")
vframes_num = 2
result = (
    bmf.encode(
        video,
        None,
        {
            "push_output": 1,
            "vframes": vframes_num,
            "format": "image2pipe",
            "avio_buffer_size": 65536, #16*4096
            "video_params": {
                "codec": "jpg",
                "width": 640,
                "height": 480
            },
        }
    )
    .start()
)
write_num = 0
for i, packet in enumerate(result):
    avpacket = packet.get(bmf.BMFAVPacket)
    data = avpacket.data.numpy()
    if write_num < vframes_num:
        output_path = "./simple_image" + str(write_num)+ ".jpg"
        write_num = write_num + 1
        with open(output_path, "wb") as f:
            f.write(data)
```

Run and check the result:

```bash
python3 image_transcoding.py
ffprobe simple_image0.jpg
rm -rf simple_image*.jpg
```
