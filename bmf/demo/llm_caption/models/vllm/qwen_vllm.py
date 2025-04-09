from models.vllm.base_vllm_model import BaseVLLMVisionModel
from abc import ABC, abstractmethod
from utils.timer import timer
from vllm import LLM, SamplingParams

class Qwen2_VL_VLLM(BaseVLLMVisionModel):
    def __init__(self, batch_size):
        super().__init__(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            batch_size=batch_size,
            prompt_format = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + \
            "<|im_start|>user\n<|vision_start|>{i}<|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n",
            image_embed="<|image_pad|>",
            image_prompt=" Do not repeat the prompt, these images are frames of a video, what do they depict? Include as much detail as possible, do not talk about in frames and structure your response with 'The video depicts'",
            title_prompt="Create a title for a video with this summary: ",
            summary_prompt="Summarise in detail what happens in this video summary: "
        )

class Qwen2_5_VL_3b_VLLM(BaseVLLMVisionModel):
    def __init__(self, batch_size):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            batch_size=batch_size,
            prompt_format = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + \
            "<|im_start|>user\n<|vision_start|>{i}<|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n",
            image_embed="<|image_pad|>",
            image_prompt=" Do not repeat the prompt, these images are frames of a video, what do they depict? Include as much detail as possible, do not talk about in frames and structure your response with 'The video depicts'",
            title_prompt="Create a title for a video with this summary: ",
            summary_prompt="Summarise in detail what happens in this video summary: "
        )
class Qwen2_5_VL_7b_VLLM(BaseVLLMVisionModel):
    def __init__(self, batch_size):
        super().__init__(
            model_name="Qwen/Qwen2.5-VL-7B-Instruct",
            batch_size=batch_size,
            prompt_format = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" + \
            "<|im_start|>user\n<|vision_start|>{i}<|vision_end|>{q}<|im_end|>\n<|im_start|>assistant\n",
            image_embed="<|image_pad|>",
            image_prompt=" Do not repeat the prompt, these images are frames of a video, what do they depict? Include as much detail as possible, do not talk about in frames and structure your response with 'The video depicts'",
            title_prompt="Create a title for a video with this summary: ",
            summary_prompt="Summarise in detail what happens in this video summary: ",
            max_model_len=16384
        )
