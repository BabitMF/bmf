/** \page TranscodeOption 转码 options

在这个例子里，使用了 options 参数进行转码。如果需要参考完整代码，请看 \ref test_transcode.py

首先，创一个graph：

**Python**
```python
import bmf

graph = bmf.graph()
```
**C++**
```cpp
#include "builder.hpp"
#include "nlohmann/json.hpp"

nlohmann::json graph_para = {
    {"dump_graph", 1}
};

auto graph = bmf::builder::Graph(bmf::builder::NormalMode, bmf_sdk::JsonParam(graph_para));
```

再进行解码：

**Python**
```python
video = graph.decode({
    "input_path": input_video_path,
    "start_time": 2
})
```
**C++**
```cpp
nlohmann::json decode_para = {
    {"input_path", "../files/img.mp4"},
    {"start_time", 2}
};
auto video = graph.Decode(bmf_sdk::JsonParam(decode_para));
```

之后，再做转码：

**Python**
```python
bmf.encode(
    video['video'],
    video['audio'],
    {
        "output_path": output_path,
        "video_params": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "preset": "fast",
            "crf": "23",
            "x264-params": "ssim=1:psnr=1"
        },
        "audio_params": {
            "codec": "aac",
            "bit_rate": 128000,
            "sample_rate": 44100,
            "channels": 2
        },
        "mux_params": {
            "fflags": "+igndts",
            "movflags": "+faststart+use_metadata_tags",
            "max_interleave_delta": "0"
        }
    }
).run()
```
**C++**
```cpp
nlohmann::json encode_para = {
    {"output_path", "./option.mp4"},
    {"video_params", {
        {"codec", "h264"},
        {"width", 1280},
        {"height", 720},
        {"crf", 23},
        {"preset", "fast"},
        {"x264-params", "ssim=1:psnr=1"}
    }},
    {"audio_params", {
        {"codec", "aac"},
        {"bit_rate", 128000},
        {"sample_rate", 44100},
        {"channels", 2}
    }},
    {"mux_params", {
        {"fflags", "+igndts"},
        {"movflags", "+faststart+use_metadata_tags"},
        {"max_interleave_delta", "0"}
    }}
};

video["video"].EncodeAsVideo(video["audio"], bmf_sdk::JsonParam(encode_para));

graph.Run();
```

以上的option里使用了一些FFMPEG选项。
