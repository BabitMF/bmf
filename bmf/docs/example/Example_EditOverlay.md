/** \page EditOverlay 编辑: Overlay

这个例子使用了 \ref video_overlay.py 里的```video_overlay``` subgraph，把一个视频 overlay 在另一个视频上。

这模块的操作跟其他Python模块一样用法：

```python
output_streams = (
    bmf.module([video['video'], logo_1], 'video_overlay', overlay_option)
)
```

如果需要完整代码，请参考 \ref test_edit.py


## Overlay Subgraph 模块

就像其他subgraph模块，overlay模块需要把input streams名称输入Subgraph的```self.inputs```：

```python
# create source stream
self.inputs.append('source')
```

自后创建overlay streams和前处理源层视频： 

```python
# create overlay stream
overlay_streams = []
for (i, _) in enumerate(option['overlays']):
    self.inputs.append('overlay_' + str(i))
    overlay_streams.append(self.graph.input_stream('overlay_' + str(i)))

# pre-processing for source layer
info = option['source']
output_stream = (
    source_stream.scale(info['width'], info['height'])
        .trim(start=info['start'], duration=info['duration'])
        .setpts('PTS-STARTPTS')
)
```

以下步骤是把logo视频overlay到源层视频上：

```python
# overlay processing
for (i, overlay_stream) in enumerate(overlay_streams):
    overlay_info = option['overlays'][i]

    # overlay layer pre-processing
    p_overlay_stream = (
        overlay_stream.scale(overlay_info['width'], overlay_info['height'])
            .loop(loop=overlay_info['loop'], size=10000)
            .setpts('PTS+%f/TB' % (overlay_info['start']))
    )

    # calculate overlay parameter
    x = 'if(between(t,%f,%f),%s,NAN)' % (overlay_info['start'],
                                            overlay_info['start'] + overlay_info['duration'],
                                            str(overlay_info['pox_x']))
    y = 'if(between(t,%f,%f),%s,NAN)' % (overlay_info['start'],
                                            overlay_info['start'] + overlay_info['duration'],
                                            str(overlay_info['pox_y']))
    if overlay_info['loop'] == -1:
        repeat_last = 0
        shortest = 1
    else:
        repeat_last = overlay_info['repeat_last']
        shortest = 1

    # do overlay
    output_stream = (
        output_stream.overlay(p_overlay_stream, x=x, y=y,
                                repeatlast=repeat_last)
    )
```

Overlay处理完毕后，就完成graph的构建。

```python
# finish creating graph
self.output_streams = self.finish_create_graph([output_stream])
```
