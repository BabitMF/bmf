# BMF 例子

## BMF模式使用

| 例子 | 描述
| --- | ---
| [普通模式的使用](./example/Example_NormalUse.md)      | 简单解码和转码例子 \ref test_transcode.py
| [Subgraph模块的使用](./example/Example_Subgraph.md)   | Subgraph模块例子 \ref test_subgraph.py , \ref subgraph_module.py
| [同步模式的使用](./example/Example_SyncMode.md)       | 同步调用模块进行解码，scale和转码 \ref test_sync_mode.py
| [生成器模式的使用](./example/Example_GeneratorMode.md) | 生成器模块scale操作 \ref test_generator.py
| [预加载模式的使用](./example/Example_PreModule.md)    | 使用预创建模块 \ref test_pre_module.py , \ref analysis.py
| [配置文件运行方式的使用](./example/Example_RunConfig.md) | 使用配置文件运行 \ref test_run_by_config.py , \ref graph.json
| [预构建模式的使用](./example/Example_ServerMode.md)   | 预构建模式：<br>i. 视频处理 <br>ii. 图像处理 <br>iii. 多视频处理 <br>iv. Filter条件 <br>v. 不从解码器模块开始 <br> \ref test_server.py


## 自定义模块的开发和使用

| 例子 | 描述
| --- | ---
| [C Module](./example/Example_CModule.md)          | 使用C++自定义模块 <br>Module header example \ref copy_module.h , Module code example \ref copy_module.cc <br>i. 复制video <br>ii. 复制image <br> \ref test_video_c_module.py
| [Python Module](./example/Example_PythonModule.md)     | 使用Python自定义模块 <br>\ref test_customize_module.py , \ref my_module.py


## 应用场景

### 编辑

| 例子 | 描述
| --- | ---
| [视频 overlay](./example/Example_EditOverlay.md)  | 主视频上叠加一个LOGO \ref test_edit.py , \ref video_overlay.py
| [视频 concat](./example/Example_EditConcat.md)    | 三段视频在时域上首尾拼接 \ref test_edit.py , \ref video_concat.py
| 音频 mix              | 两段音频进行混音操作 \ref test_edit.py , \ref audio_mix.py
| 比较复杂的编辑例子        | 三段视频分别加LOGO并首尾拼接 \ref test_edit.py

### 转码

| 例子 | 描述
| --- | ---
| [Audio 转码](./example/Example_TranscodeAudio.md)            | 简单音频转码案例 \ref test_transcode.py
| [Image 转码](./example/Example_TranscodeImage.md)            | 简单图像转码案例 \ref test_transcode.py
| [Video 转码](./example/Example_TranscodeVideo.md)            | 典型视频转码案例（添加两个LOGO并添加前后贴片 \ref test_transcode.py
| Null audio           | \ref test_transcode.py
| 使用 callback         | \ref test_transcode.py
| HLS转码               | \ref test_transcode.py
| 加密流的转码           | \ref test_transcode.py
| [转码参数](./example/Example_TranscodeOption.md)          | 编解码参数案例 \ref test_transcode.py
| 视频和音频 concat     | \ref test_transcode.py

### 人面检测

| 例子 | 描述
| --- | ---     
| [Face Detection](./example/Example_FaceDetect.md)        | Python人面检测 \ref detect_sample.py , \ref onnx_face_detect.py <br> \ref test_server_detect.py


### 实现超分

| 例子 | 描述
| --- | ---     
| [Prediction](./example/Example_Predict.md)               | Python实现超分模块并转码 \ref predict_sample.py , \ref onnx_sr.py
