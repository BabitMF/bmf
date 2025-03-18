# README
This demo illustrates the use of deepseek-vl2 for captioning (describing) a video input.
It first extract frames from an input video at 1 fps, converts them into PIL format and offline inferences them in batches of `batch_size`. The result is written to a json file, and each individual description of each batch is further inferenced again to give a summary and a title.

## Requirements
- NOTE this demo requires CUDA support
- It may be possible to replace `to.("cuda")` and `.cuda()` calls with `.to("cpu")` for cpus or `.to("mps")` for apple cpus but more modifications will be needed elsewhere

## Installation

Tested with Debian 10 and python 3.8.2 and 3.10.0 version and Nvidia L4 GPU.

1. Install ffmpeg
```
apt install ffmpeg
```
2. Install bmf library with pip
```
pip install BabitMF-GPU
```
3. Clone deepseek-vl2 repo
```
git clone https://github.com/deepseek-ai/DeepSeek-VL2.git
```
4. Install model driver, cd into repo and pip install dependencies. Use `pip install .` if on 3.10.0
```
(cd DeepSeek-VL2 && pip install -e .)
```
5. Install the model outside the cloned repo, using `deepseek-ai/deepseek-vl2-tiny` in this case
```
git clone https://huggingface.co/deepseek-ai/deepseek-vl2-tiny
```
6. For Python version >=3.10.0 only: to resolve Numpy is not available runtime error, additionally run
```
pip install 'numpy<2'
```
7. Run the demo after changing input to file path of a video
```
python test_llm_caption.py
```
## Configuration 
- `batch_size`: specifies how many images to be attached to a single prompt, `4` by default
- `result_path`: specifies where the json file will be stored, `result.json` by default in current working directory
- `multithreading`: specifies if the model should do inferences with multiple threads - `false` by defualt. Setting to be `true` and `max_threads: 1` give similar performance to setting `multithreading: false` / omitting this option
- `max_threads`: specifies how many threads can inference concurrently - requires lots of memory when using multiple threads with large `batch_size` and starts to block for a thread to finish when `max_threads` is reached
- `pass_through`: `false` by default, specifies whether output packets from a previous module should be propagated / passed through this module to preserve the output

## Output

Note that the last batch may not have the same number of frames as `batch_size` due to remainders in imperfect division

A json file named `result.json` by default will be created in the current working directory with the following schema:
```json
{
    "video_title": "<title>",
    "batch_size": <size>,
    "batches": [
        {
            "batch_id": <1>,
            "frames": <number>,
            "description": "<description>",
        },
        {
            ...
        }
    ],
    "frames_analysed": <amount>,
    "summary": "<summary>"
}
```
Summary for `big_bunny_1min_30fps.mp4`:

The images depict a serene landscape with trees, grass, and a small stream. They depict a scene from a video where a large, dark-colored creature is sleeping inside a grassy mound with a tree growing out of it. The creature appears to be resting or sleeping, and the environment is lush and green, suggesting a natural or forest setting. These images depict a bear resting in a grassy area with rocks and greenery around it. These images depict a scene from an animated film or series featuring anthropomorphic animals in a natural setting. The character appears to be a large, rotund rabbit with long ears, standing in front of a tree with its arms raised in a celebratory or triumphant gesture. The background shows a lush, green landscape with grass, rocks, and trees, suggesting a peaceful and idyllic environment. The lighting indicates it might be either dawn or dusk, adding a warm and serene atmosphere to the scene.
## Performance

Running on Nvidia L4 with 24GB memory:

- Single threaded on `big_bunny_1min_30fps.mp4` resulted in out of memory on batch sizes 10 and bigger.
- Two threaded on `big_bunny_1min_30fps.mp4` resulted in out of memory on batch sizes 5 and greater. Performance was slower than single threaded due to context switches and GIL
- Four threaded resulted in out of memory on any batch size
