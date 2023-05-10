# broadcaster demo

[English Doc](./readme.md)

## 导播demo的graph组成

![](./broadcaster.png)

- wallclock module
    wallclock用于均匀生成音视频时戳，比如导播配置为fps 25帧，44100双声道lc-aac输出，wallclock生成每一帧音视频的时戳，作为背景音视频时戳输入给到streamhub模块，同时作为稳定源驱动整个graph，使导播App源源不断输出流

- ffmpeg_decoder module

    导播的素材源，它能生成音视频流。本demo中使用了bmf内置的ffmpeg_decoder module, demo中支持rtmp输入流，且音频为44100双声道的lc-aac源。该module对接streamhub模块。
    
- streamhub module

    内部对每路输入流都有jitter buffer，给多路流设置缓存，使其能均匀化输出，多流混音混屏需要对齐时间戳和帧率，streamhub将所有的输入流帧根据背景音视频时戳绑定起来，作为一个整体输出到混流混音模块

- audiomix module

    混音模块，初始化时可配置导播音频输出参数，audiomix模块将streamhub输出的framelist进行混音，只支持1024 sample，双声道，44100采样的音频输入

- videolayout module

    videolayout根据streamhub输出的framelist混屏，可配置不同的混屏参数，控制多路流的显示坐标。

- ffmpeg_encoder module

    导播输出模块,对接videolayout和audiomix，编码输出rtmp流


## 运行导播demo需要的步骤

1. 一个可推拉流的rtmp服务器，用作导播流输出
2. 安装bmf python包
3. python3 broadcaster.py

## 通过http接口控制导播app

每个素材源有唯一的一个`index`用来标识它的id，控制接口通过该`index`来表示控制哪个素材源。

- 增加素材源

    ```
    curl -X POST -d '{"method":"add_source", "index":0, "input_path":"rtmp://localhost/live/zx"}' http://localhost:55566/
    ```


- 删除素材源

    ```
    curl -X POST -d '{"method":"remove_source", "index":0}' http://localhost:55566/
    ```


- 改变视频布局

    ```
    curl -X POST -d '{"method":"set_layout", "layout":{"background_color":"#958341"}}' http://localhost:55566/

    curl -X POST -d '{"method":"set_layout", "layout":{"layout_mode":"speaker"}}' http://localhost:55566/

    curl -X POST -d '{"method":"set_layout", "layout":{"layout_mode":"gallery"}}' http://localhost:55566/
    ```

- 设置素材源音频音量

    音量范围是[0,10]

    ```
    curl -X POST -d '{"method":"set_volume", "index":0, "volume":1.5}' http://localhost:55566/
    ```
