# Bmf caption demo
This is a c++ caption tensorrt inference demo, model is llava-v1.6, caption means use llm to describe the video.
## Download
Building this demo requires several third-party libraries such as tensorrt_llm and libtorch. You can download these libraries by running the download.sh script. And this demo also require TensorRT10, you should download TensorRT10 party from NVIDIA to run this demo.
```
./download.sh
```
Our llava convert and inference demo use this model in hugging-face, firstly you should download these model
```
git-lfs clone https://huggingface.co/Trelis/llava-v1.6-mistral-7b-PATCHED
git-lfs clone https://huggingface.co/openai/clip-vit-large-patch14-336
```
Then, convert vit and llava model so that you can utilize TensorRT and TensorRT-LLM to achieve high performance inference, we have provided a script to converting these models
```
cd convert
./build.sh /path/to/llava-v1.6-mistral-7b-PATCHED /path/to/openai/clip-vit-large-patch14-336 /path/to/dst llava_py
cp ../table.raw /path/to/dst/.
```
## Build
Use the build_caption_demo.sh script to compile.
```
./build_caption_demo.sh
```

## Inference
Now you can use this caption demo to describe your video by run.sh script.
```
./run.sh
```