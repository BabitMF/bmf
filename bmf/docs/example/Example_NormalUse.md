/** \page NormalUse 基本例子

这页描述BMF基本用法。开始时需要构建一个graph：

```python
import bmf

graph = bmf.graph()
```

Graph初始化之后，解码input video：

```python
video = graph.decode({
    "input_path": input_video_path
})
```

把之前解码的分出 ```video['video']``` 和 ```video['audio']``` 做转码，```run```会开始validate和执行整个graph的构建和执行，完成后输出一个video file：

```python
bmf.encode(
    video['video'],
    video['audio'],
    {
        "output_path": output_path,
        "video_params": {
            "codec": "h264",
            "width": 320,
            "height": 240,
            "crf": 23,
            "preset": "veryfast"
        },
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

如果需要完整代码，可以参考 \ref test_transcode.py
