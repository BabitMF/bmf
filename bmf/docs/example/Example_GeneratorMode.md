/** \page GeneratorMode Generator Mode

生成器方式以普通异同，主要是运用```start()```（一般用的是```run()```）：

```python
frames = (
    bmf.graph()
        .decode({'input_path': "../files/img.mp4"})['video']
        .ff_filter('scale', 299, 299)  # or you can use '.scale(299, 299)'
        .start()  # this will return a packet generator
)
```

生成后的frames，能像iterator一样使用：

```python
for i, frame in enumerate(frames):
    # convert frame to a nd array
    if frame is not None:
        np_frame = frame.to_ndarray(format='rgb24')

        # we can add some more processing here, e.g. predicting
        print('frame', i, 'shape', np_frame.shape)
    else:
        break
```

如果需要完整代码，可以参考 \ref test_generator.py
