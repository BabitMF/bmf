/** \page TranscodeVideo 视频转码

这个例子描述如何做简单的视频转码。如果需要参考完整代码，请看 \ref test_transcode.py (Python) 或 \ref c_transcode.cpp (C++)

首先，创一个graph：

**Python**
```python
import bmf

my_graph = bmf.graph()
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
# tail video
tail = my_graph.decode({'input_path': input_video_path_1})

# header video
header = my_graph.decode({'input_path': input_video_path_2})

# main video
video = my_graph.decode({'input_path': input_video_path_3})

# logo video
logo_1 = (
    my_graph.decode({'input_path': logo_video_path_1})['video']
        .scale(logo_width, logo_height)
)
logo_2 = (
    my_graph.decode({'input_path': logo_video_path_2})['video']
        .scale(logo_width, logo_height)
        .ff_filter('loop', loop=-1, size=991)
        .ff_filter('setpts', 'PTS+3.900/TB')
)
```
**C++**
```cpp
nlohmann::json decode_tail_para = {
    {"input_path", "../files/header.mp4"}
};
auto tail = graph.Decode(bmf_sdk::JsonParam(decode_tail_para));

nlohmann::json decode_header_para = {
    {"input_path", "../files/header.mp4"}
};
auto header = graph.Decode(bmf_sdk::JsonParam(decode_header_para));

nlohmann::json decode_main_para = {
    {"input_path", "../files/img.mp4"}
};
auto video = graph.Decode(bmf_sdk::JsonParam(decode_main_para));

nlohmann::json decode_logo1_para = {
    {"input_path", "../files/xigua_prefix_logo_x.mov"}
};
auto logo_1 = graph.Decode(bmf_sdk::JsonParam(decode_logo1_para))["video"]
    .Scale("320:144");

nlohmann::json decode_logo2_para = {
    {"input_path", "../files/xigua_loop_logo2_x.mov"}
};
auto logo_2 = graph.Decode(bmf_sdk::JsonParam(decode_logo2_para))["video"]
    .Scale("320:144")
    .Loop("loop=-1:size=991")
    .Setpts("PTS+3.900/TB");
```

主视频将按比例缩放并覆盖logo：

**Python**
```python
# main video processing
main_video = (
    video['video'].scale(output_width, output_height)
        .overlay(logo_1, repeatlast=0)
        .overlay(logo_2,
                    x='if(gte(t,3.900),960,NAN)',
                    y=0,
                    shortest=1)
)
```
**C++**
```cpp
auto main_video = video["video"].Scale("1280:720")
    .Overlay({logo_1}, "repeatlast=0")
    .Overlay({logo_2}, "shortest=1:x=if(gte(t,3.900),960,NAN):y=0");
```

之后，将三个视频连接在一起：

**Python**
```python
# concat video
concat_video = (
    bmf.concat(header['video'].scale(output_width, output_height),
                main_video,
                tail['video'].scale(output_width, output_height),
                n=3)
)
```
**C++**
```cpp
auto concat_video = graph.Concat({
    header["video"].Scale("1280:720"),
    main_video,
    tail["video"].Scale("1280:720")
}, "n=3");
```

同样的，也把音频连接在一起：

**Python**
```python
# concat audio
concat_audio = (
    bmf.concat(header['audio'],
                video['audio'],
                tail['audio'],
                n=3, v=0, a=1)
)
```
**C++**
```cpp
auto concat_audio = graph.Concat({
    header["audio"],
    video["audio"],
    tail["audio"]
}, "a=1:n=3:v=0");
```

最终做转码：

**Python**
```python
bmf.encode(concat_video,
    concat_audio,
    {
        "output_path": output_path,
        "video_params": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "preset": "veryfast",
            "crf": "23",
            "x264-params": "ssim=1:psnr=1"
        },
        "audio_params": {
            "codec": "aac",
            "bit_rate": 128000,
            "sample_rate": 48000,
            "channels": 2
        },
        "mux_params": {
            "fflags": "+igndts",
            "movflags": "+faststart+use_metadata_tags",
            "max_interleave_delta": "0"
        }
    })
.run()
```
**C++**
```cpp
nlohmann::json encode_para = {
    {"output_path", "./video.mp4"},
    {"video_params", {
        {"codec", "h264"},
        {"width", 1280},
        {"height", 720},
        {"crf", 23},
        {"preset", "veryfast"},
        {"x264-params", "ssim=1:psnr=1"},
        {"vsync", "vfr"},
        {"max_fr", 60}
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

graph.Encode(concat_video, concat_audio, bmf_sdk::JsonParam(encode_para));

graph.Run();
```
