from models.vllm.base_vllm_model import BaseVLLMVisionModel
from abc import ABC, abstractmethod
from utils.timer import timer
from vllm import LLM, SamplingParams

class LLaVA_Next_Video_VLLM(BaseVLLMVisionModel):
    def __init__(self, batch_size):
        super().__init__(
            model_name="llava-hf/LLaVA-NeXT-Video-7B-hf",
            batch_size=batch_size,
            prompt_format="USER: {i}\n{q} ASSISTANT:",
            image_embed="<image>",
            image_prompt="These images are frames of a video, what do they depict? Begin with 'The video'",
            title_prompt="Create a short title of a video with this summary: ",
            summary_prompt="The text describes a video, explain in detail what happens: ",
            overrides={"architectures": ["LlavaNextVideoForConditionalGeneration"]}
        )

class LLaVA_One_Vision_VLLM(BaseVLLMVisionModel):
    def __init__(self, batch_size):
        super().__init__(
            model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf",
            batch_size=batch_size,
            prompt_format="<|im_start|>user{i}\n{q}<|im_end|><|im_start|>assistant\n",
            image_embed="<image>",
            image_prompt="These images are frames of a video, what do they depict? Do not structure your response by frame. Begin with 'The video'",
            title_prompt="Create a short title of a video with this summary: ",
            summary_prompt="The text describes a video, explain in detail what happens: ",
        )
