import bmf
import py_deoldify_module

# Add the DeOldify folder to the python search path
import sys
sys.path.insert(0, '/content/DeOldify')
print(sys.path)

input_video_path = '/content/drive/MyDrive/deoldify/oldvideo.mp4'
output_video_path = '/content/colored_video.mp4'
model_weight_path = '/content/DeOldify/'

graph = bmf.graph()

video = graph.decode({"input_path": input_video_path})

output_video = video['video'].module('py_deoldify_module', option={"model_path": "/content/DeOldify/"})

bmf.encode(
    output_video[0],
    None,
    {"output_path": output_video_path}
    ).run()