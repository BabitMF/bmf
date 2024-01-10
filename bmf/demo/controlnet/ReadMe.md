# BMF ControlNet Demo

This demo shows how to use ControlNet+StableDiffusion to generate image from text prompts in BMF. We use a performance-optimized ControlNet [implementation](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023/controlnet). This implementation accelerates the canny2image app in the official ControlNet repo.

You need to compile or install bmf before running the demo. Please refer to the [document](https://babitmf.github.io/docs/bmf/getting_started_yourself/install/) on how to build or install bmf.

### Generate TensorRT Engine

First we need to put the ControlNet code in the demo directory. This repo contains lots of samples of TensorRT, the ControlNet implementation we need in located in `trt-samples-for-hackathon-cn/Hackathon2023/controlnet`
```Bash
git clone https://github.com/NVIDIA/trt-samples-for-hackathon-cn.git
# copy the controlnet implementation to the demo path for simplicity
cp -r trt-samples-for-hackathon-cn/Hackathon2023/controlnet bmf/demo/controlnet
```

Download the state dict from HuggingFace and generate the TensorRT engine. You need to change the state dict path in `controlnet/export_onnx.py:19` to where you put the file. Then run `preprocess.sh` to build the TensorRT engine.
```Bash
cd bmf/demo/controlnet/controlnet/models
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/models/control_sd15_canny.pth
# Change the path to './models/control_sd15_canny.pth' in controlnet/export_onnx.py:19
cd .. # go back to the controlnet directory
bash preprocess.sh
```

Once the script runs successfully, several `.trt` files will be generated, which are the TensorRT engines. Copy the generated TensorRT engines to the directory of the demo and run the ControlNet pipeline using the `test_controlnet.py` script
```Bash
mv *.trt path/to/the/demo
cd path/to/the/demo
python test_controlnet.py
```
The pipeline will generate a new image based on the input image and prompt.