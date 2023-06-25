import bmf
import py_deoldify_module

# Add the DeOldify folder to the python search path
import sys
sys.path.insert(0, './DeOldify')
print(sys.path)

input_video_path = './DeOldify/test_videos/test_video.mp4'
output_video_path = 'colored_video.mp4'
model_weight_path = './DeOldify/'

graph = bmf.graph()

video = graph.decode({"input_path": input_video_path})

output_video = video['video'].module('py_deoldify_module', option={"model_path": model_weight_path})

bmf.encode(
    output_video[0],
    None,
    {"output_path": output_video_path}
    ).run()