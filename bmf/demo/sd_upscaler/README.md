# Wrap the Hugging Face diffusor into a BMF module


- This demo demonstates how to use BMF to develop a video super resolution application using the state of art stable diffusion model. It must be run in a NVidia GPU environment. As you know, the stable diffusion model is computational expensive. It runs very slow if you're using a low end GPU.

- Original model and reference code: https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler


- Installation steps:
  - Install the diffusers package
  ```Bash
    pip install diffusers transformers accelerate scipy safetensors
  ```
  - Install pillow and numpy package
  ```Bash
    pip install pillow numpy
  ```




