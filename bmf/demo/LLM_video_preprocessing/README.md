It's a demo for LLM video/image generating training preprocessing.

Based on BMF, it's flexible to build and integrate algorithms into whole pipeline of preprocessing.

Two part of them are included:
1. Clip processing
The input video will be split according to scene change, and subtitles in the video will be detected and cropped by OCR module, and the video quality will be assessed by BMF provided aesthetic module.
After that, the finalized video clips will be encoded as output.
## Prerequisites
Please pip install all the dependencies in `requirement.txt`
## Run
```
python main.py --input_file <your test video>
```
There will be output info and clips to be stored in `clip_output` of current path.

2. Caption
Please reference the README in "bmf/bmf/demo/fast_caption_module" 