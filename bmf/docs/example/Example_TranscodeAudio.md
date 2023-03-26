/** \page TranscodeAudio 音频转码

这个例子描述如何做简单的音频转码。如果需要参考完整代码，请看 \ref test_transcode.py

首先，创一个graph：

```python
graph = bmf.graph()
```

再进行解码：

```python
video = graph.decode({
    "input_path": input_video_path
})
```

之后，再做转码：

```python
bmf.encode(
    None,
    video['audio'],
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
```
