/** \page FaceDetect 人脸检测

这个例子使用ONNX模型对视频流执行人脸检测之后，将结果叠加到视频上显示。例子使的是 \ref onnx_face_detect.py 里的```onnx_face_detect``` 模块。

模块构建如下：

```python
onnx_face_detect = bmf.create_module('onnx_face_detect', {
    "model_path": "version-RFB-640.onnx",
    "label_to_frame": 1
})
```

这个例子首先会下载视频，然后在做解码和用onnx_face_detect模块：

```python
# Download video file from URL
video_stream = graph.download({
    'input_url': 'https://github.com/fromwhzz/test_video/raw/master/face.mp4',
    'local_path': '../files/face_test.mp4'
}).decode()

# Using onnx_face_detect module
detect_stream = video_stream['video'].module('onnx_face_detect', pre_module=onnx_face_detect)
```

如果需要完整代码，请参考 \ref detect_sample.py


## 人脸检测模块

 ```init``` 时，会使用ONNX Runtime加载并运行ONNX模型：

```python
# load model
self.sess_ = rt.InferenceSession(self.model_path_)
```

数据将被接收和缓存：

```python
while not input_queue.empty():
    pkt = input_queue.get()
    if pkt.get_timestamp() == Timestamp.EOF:
        # we should do all frames processing in following loop
        self.eof_received_ = True
    if pkt.get_data() is not None:
        self.frame_cache_.put(pkt.get_data())
```

最后，数据将通过面部检测算法运行：

```python
# detect processing
while self.frame_cache_.qsize() >= self.in_frame_num_ or \
        self.eof_received_:
    data_list, extra_data_list = self.detect()
    for index in range(len(data_list)):
        # add sr output frame to task output queue
        pkt = Packet()
        pkt.set_timestamp(data_list[index].pts)
        pkt.set_data(data_list[index])
        task.get_outputs()[0].put(pkt)
        # push output
        if(output_queue_size>=2):
            pkt = Packet()
            pkt.set_timestamp(data_list[index].pts)
            pkt.set_data(extra_data_list[index])
            task.get_outputs()[1].put(pkt)

    # all frames processed, quit the loop
    if self.frame_cache_.empty():
        break
```

在 ```detect``` 里，每当检测到脸部时，将对frame进行处理并用边框框覆盖：

```python
def detect(self):
    frame_num = min(self.frame_cache_.qsize(), self.in_frame_num_)
    input_frames = []
    input_pil_arrays = []
    if frame_num==0:
        return [],[]
    for i in range(frame_num):
        frame = self.frame_cache_.get()
        input_frames.append(frame)
        input_pil_arrays.append(frame.to_image())

    input_tensor = self.pre_process(input_pil_arrays)
    scores,boxes = self.sess_.run(self.output_names_, {self.input_names_[0]: input_tensor})
    detect_result_list= self.post_process(input_pil_arrays,boxes,scores)
    if self.label_frame_flag_==1:
        result_frames=self.label_frame(input_frames,input_pil_arrays,detect_result_list)
        return result_frames,detect_result_list
```
