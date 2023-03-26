/** \page TranscodeImage 图像转码

这个例子描述如何做简单的图像scale操作和转码。如果需要参考完整代码，请看 \ref test_transcode.py (Python) 或 \ref c_transcode.cpp (C++)

首先，创一个graph再进行解码：

**Python**
```python
import bmf

graph = bmf.graph()
    .decode({'input_path': input_video_path})['video']
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

在这个例子里，图像将按比例缩小到 320x240：
**Python**
```python
.scale(320, 240)
```
**C++**
```cpp
.Scale("320:240")
```

之后，再做转码：
**Python**
```python
.encode(None, {
    "output_path": output_path,
    "video_params": {
        "codec": "jpg",
        "width": 320,
        "height": 240
    }
}).run()
```
**C++**
```cpp
nlohmann::json encode_para = {
    {"output_path", "./image.jpg"},
    {"format", "mjpeg"},
    {"video_params", {
        {"codec", "jpg"},
        {"width", 320},
        {"height", 240}
    }}
};

video["video"].EncodeAsVideo(video["audio"], bmf_sdk::JsonParam(encode_para));

graph.Run();
```