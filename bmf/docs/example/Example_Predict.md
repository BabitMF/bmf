/** \page Predict 超分

这个例子使用ONNX超分模型执行实现超分，然后对视频进行转码。例子使的是 \ref onnx_sr.py 里的```onnx_sr``` 模块。

模块构建如下，能让模块仅初始化一次，并且可以重复使用：

```python
sr_mod = bmf.create_module('onnx_sr', {
    "model_path": "v1.onnx"
})
```

构建后，BMF会进行解码，超分，最后做转码：

```python
bmf.graph()
    .decode({'input_path': "../files/img_s.mp4"})['video']
    .module('onnx_sr', pre_module=sr_mod)
    .encode(None, {"output_path": "../files/out.mp4"})
    .run()
```

如果需要完整代码，请参考 \ref predict_sample.py

## 超分模块

在模块里 ```init``` 时，会使用ONNX Runtime加载并运行ONNX模型：

```python
self.sess_ = rt.InferenceSession(self.model_path_)
```

数据将被接收和缓存：

```python
# add all input frames into frame cache
while not input_queue.empty():
    pkt = input_queue.get()
    if pkt.get_timestamp() == Timestamp.EOF:
        # we should done all frames processing in following loop
        self.eof_received_ = True
    if pkt.get_data() is not None:
        self.frame_cache_.put(pkt.get_data())
```

最后，数据将通过超分算法运行：

```python
# sr processing
while self.frame_cache_.qsize() >= self.in_frame_num_ or \
        self.eof_received_:
    if self.frame_cache_.qsize() > 0:
        for frame in self.sr_process():
            # add sr output frame to task output queue
            pkt = Packet()
            pkt.set_timestamp(frame.pts)
            pkt.set_data(frame)
            output_queue.put(pkt)
            
    # all frames processed, quit the loop
    if self.frame_cache_.empty():
        break
```

在```sr_process```函数内前处理后，主要超分部分如下：

```python
# predict
output_tensor = self.sess_.run([self.output_name_], {self.input_name_: input_tensor})
```

后处理部分，把video frame转成yuv420p再输出：

```python
# create output frames
out_frames = []
for i in range(self.out_frame_num_):
    # convert nd array to video frame and convert rgb to yuv
    out_frame = VideoFrame.from_ndarray(output_tensor[i], format='rgb24').reformat(format='yuv420p')

    if self.frame_cache_.empty():
        break

    # dequeue input frame
    # copy frame attributes from (2 * i + 1)th input frame
    input_frame = self.frame_cache_.get()
    out_frame.pts = input_frame.pts
    out_frame.time_base = input_frame.time_base
    out_frames.append(out_frame)
```
