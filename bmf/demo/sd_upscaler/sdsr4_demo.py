import bmf
import py_sdsr4_module

# The test clip can be downloaded from:
# https://github.com/BabitMF/bmf/releases/download/files/files.tar.gz
# You can modify the input and output directory according to your needs

input_video_path = './test/low_res_input.mp4'

output_video_path = './test/upscaled_video_x4.mp4'

# This will download the models from huggingface's website
# You can save it in your filesystem and specify the model path.
model_path = 'stabilityai/stable-diffusion-x4-upscaler'

graph = bmf.graph()

video = graph.decode({"input_path": input_video_path})

output_video = video['video'].module('py_sdsr4_module', option={'model_path': model_path})

bmf.encode(output_video[0], None, {'output_path': output_video_path}).run()
