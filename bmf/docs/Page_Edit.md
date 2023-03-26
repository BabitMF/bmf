/** \page Edit 编辑

## 编辑场景Solution案例

### 场景特性

编辑场景下的case具有以下特性：

- Pipeline复杂度较高，DAG结构由数量较多的节点组成
- DAG中存在subgraph、开发者自写模块、内置模块
- 内置模块中FIlter模块数量众多，常包含大量不同种类的filter模块
- 多输入、多输出节点很常见

编辑场景较为复杂，即便框架提供了易用性较高的流式接口，开发者在编写编辑场景的case时可能会感到不顺手，为此我们提供了部分参考文档：

- 部分编辑函数/模块改造过程中搜集的注意事项，详见：https://
- 模块的调用（创建）方式说明文档，详见：https://
- 部分模块的可选参数使用说明，详见：https://

### 痛点
编辑场景下往往节点众多，需要同时对多个源视频进行处理，且不同节点之间存在同步等待的协作需求，因此运行编辑用例时的内存占用通常较高。

### 编辑案例下的DAG展示

编辑场景下的案例众多，不同的案例的DAG也差别很大，暂且仅对部分场景下的DAG做一个展示。

1、原素材为一段主视频和一个LOGO，将LOGO叠加到主视频上的指定位置

<img src="./images/video_overlay.png" style="zoom:50%;" />

2、原素材为两段视频，将两者在时域上前后拼接，且在拼接处设置转场特效，这里的特效为两视频叠加

<img src="./images/video_concat.png" style="zoom:50%;" />

3、原素材为一段视频，将视频的最后一帧延长至指定时长

<img src="./images/loop_last_pic.png" style="zoom:50%;" />

4、多段视频、音频、图像等各自进行一系列预处理，而后拼接为整体

<img src="./images/edit_prototype.png" style="zoom:50%;" />

### 编辑函数代码示例

以上述“LOGO叠加到主视频”的case为例，展示其编辑函数。编写函数时，可以直接使用流式接口依次调用各个模块，也可以使用编写subgraph的方式。下面对两种不同的函数编写方式做一个展示。

##### 方式1（不使用subgraph）

test_video_overlays中的示例代码通过BMF应用层接口：graph(), decode(),module(), encode(), run()，创建运行一个视频叠加的pipeline

```python
def test_video_overlays(self):
    input_video_path = "../files/img.mp4"
    logo_path = "../files/xigua_prefix_logo_x.mov"
    output_path = "./overlays.mp4"

    # create graph
    my_graph = bmf.graph()

    # main video
    main_video = my_graph.decode({'input_path': input_video_path})

    # logo video
    logo_video = my_graph.decode({'input_path': logo_path})['video']

    # main video preprocess
    main_video = main_video.scale(1280, 720).trim(start=0, duration=7).setpts('PTS-STARTPTS')

    # logo video preprocess
    logo_video = logo_video.scale(300, 200).loop(loop=0, size=10000).setpts('PTS+0/TB')

    # do overlay
    output_stream = (
      main_video.overlay(logo_video, x='if(between(t,0,4),0,NAN)', y='if(between(t,0,4),0,NAN)',repeatlast=1)
    )

    # encode
    (
      output_stream[0].encode(None, {
        "output_path": output_path,
        "video_params": {
          "width": 640,
          "height": 480,
          'codec': 'h264'
        }
      }).run()
    )
    
if __name__ == '__main__':
  	test_video_overlays()
```
##### 方式2（使用subgraph）

将video overlay的实现细节封装在subgraph内部，对于需要频繁使用的一系列联合操作，建议可封装为subgraph

```python
def test_video_overlays(self):
    input_video_path = "../files/img.mp4"
    logo_path = "../files/xigua_prefix_logo_x.mov"
    output_path = "./overlays.mp4"

    # create graph
    my_graph = bmf.graph()
		
    # overlay params
    overlay_option = {
      "source": {
        "start": 0,
        "duration": 7,
        "width": 1280,
        "height": 720
      },
      "overlays": [
        {
          "start": 0,
          "duration": 4,
          "width": 300,
          "height": 200,
          "pox_x": 0,
          "pox_y": 0,
          "loop": 0,
          "repeat_last": 1
        }
      ]
    }

    # main video
    main_video = my_graph.decode({'input_path': input_video_path})['video']

    # logo video
    logo_video = my_graph.decode({'input_path': logo_path})['video']

    # call 'my_overlay' subgraph to do overlay
    output_streams = (
      bmf.module([main_video, logo_video], 'video_overlay', overlay_option)
    )
		
    # encode
    (
      output_streams[0].encode(None, {
        "output_path": output_path,
        "video_params": {
          "width": 640,
          "height": 480,
          'codec': 'h264'，
          "crf": 23,
          "preset": "veryfast"
        }
      }).run()
    )
    
if __name__ == '__main__':
  	test_video_overlays()
```

其中，subgraph ‘video overlay’的代码如下：

```python
class video_overlay(SubGraph):
    def create_graph(self, option=None):
        # create source stream
        self.inputs.append('source')
        source_stream = self.graph.input_stream('source')

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

        # finish creating graph
        self.output_streams = self.finish_create_graph([output_stream])
```

详细代码可以参考 example/edit/test_edit.py
