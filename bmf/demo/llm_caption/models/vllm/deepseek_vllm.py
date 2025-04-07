from models.vllm.base_vllm_model import BaseVLLMVisionModel
from abc import ABC, abstractmethod
from utils.timer import timer
from vllm import LLM, SamplingParams

class Deepseek_VL2_VLLM(BaseVLLMVisionModel):
    def __init__(self, batch_size):
        super().__init__(
            model_name="deepseek-ai/deepseek-vl2-tiny",
            batch_size=batch_size,
            prompt_format="<|User|>: {i}\n{q}\n\n<|Assistant|>:",
            image_embed="<image>",
            image_prompt="These images are frames of a video, what do they depict? Do NOT structure your response by frame, begin with 'The video depicts'",
            title_prompt="Respond with a short title. Create a fitting title of a video with this summary: ",
            summary_prompt="Respond with only a summary. The text describes a video, explain in detail what happens: ",
            overrides={"architectures": ["DeepseekVLV2ForCausalLM"]}
        )
