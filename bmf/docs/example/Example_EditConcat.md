/** \page EditConcat 编辑: Concat

这个例子使用了 \ref video_concat.py 里的```video_concat``` subgraph，把所视频和音频都合并在一起，每个视频之间进行过渡效果。

这模块的操作跟其他Python模块一样用法：

```python
# concat video and audio streams with 'video_concat' module
concat_streams = (
    bmf.module([
        video1['video'],
        video2['video'],
        video3['video'],
        video1['audio'],
        video2['audio'],
        video3['audio']
    ], 'video_concat', option)
)
```

如果需要完整代码，请参考 \ref test_edit.py

## Concat Subgraph 模块

就像其他subgraph模块，concat模块需要把全部input streams名称输入Subgraph的```self.inputs```。以下代码首先从```options```获取各video stream的```duration```，```transition_time```之类的讯息，在每个video stream中段进行过渡：

```python
# process video streams
for i in range(video_stream_cnt):
    # create a input stream
    stream_name = 'video_' + str(i)
    self.inputs.append(stream_name)
    video_stream = (
        self.graph.input_stream(stream_name)
            .scale(option['width'], option['height'])
    )

    if option['video_list'][i]['transition_time'] > 0 and i < video_stream_cnt - 1:
        split_stream = video_stream.split()
        video_stream = split_stream[0]
        transition_stream = split_stream[1]
    else:
        transition_stream = None

    # prepare concat stream
    info = option['video_list'][i]
    trim_time = info['duration'] - info['transition_time']
    concat_stream = (
        video_stream.trim(start=info['start'], duration=trim_time)
            .setpts('PTS-STARTPTS')
    )

    # do transition, here use overlay instead
    if prev_transition_stream is not None:
        concat_stream = concat_stream.overlay(prev_transition_stream, repeatlast=0)

    # add to concat stream
    concat_video_streams.append(concat_stream)

    # prepare transition stream for next stream
    if transition_stream is not None:
        prev_transition_stream = (
            transition_stream.trim(start=trim_time, duration=info['transition_time'])
                .setpts('PTS-STARTPTS')
                .scale(200, 200)
        )
```

然后在把video streams链接在一起：

```python
# concat videos
concat_video_stream = bmf.concat(*concat_video_streams, n=video_stream_cnt, v=1, a=0)
```

Audio streams方面也是必须链接，同样的在中段进行过渡效果：

```python
# process audio
concat_audio_stream = None
if audio_stream_cnt > 0:
    concat_audio_streams = []
    for i in range(audio_stream_cnt):
        # create an input stream
        stream_name = 'audio_' + str(i)
        self.inputs.append(stream_name)

        # pre-processing for audio stream
        info = option['video_list'][i]
        trim_time = info['duration'] - info['transition_time']
        audio_stream = (
            self.graph.input_stream(stream_name)
                .atrim(start=info['start'], duration=trim_time)
                .asetpts('PTS-STARTPTS')
                .afade(t='in', st=0, d=2)
                .afade(t='out', st=info['duration'] - 2, d=2)
        )

        # add to concat stream
        concat_audio_streams.append(audio_stream)

    # concat audio
    concat_audio_stream = bmf.concat(*concat_audio_streams, n=audio_stream_cnt, v=0, a=1)
```

Concat处理完毕后，就完成graph的构建。

```python
# finish creating graph
self.output_streams = self.finish_create_graph([concat_video_stream, concat_audio_stream])
```
