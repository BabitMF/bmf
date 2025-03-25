from models.base_model import ModelFactory
from abc import ABC, abstractmethod
from utils.timer import timer
import torch
# tested works with python3.10 and newest beta release of transformers
class LLaVA(ModelFactory, ABC):
    def __init__(self, processor_class,
                 model_class, 
                 model_path,
                 image_prompt,
                 title_prompt,
                 summary_prompt):
        super().__init__()
        # requries flash attention 2 installed and pip install bitsandbytes
        self.model = model_class.from_pretrained(model_path,
                                                 torch_dtype=torch.float16,
                                                 load_in_4bit=True,
                                                 use_flash_attention_2=True).to("cuda")
        self.processor = processor_class.from_pretrained(model_path) 
        self.prompt_format = [
            {
            "role": "user",
            "content": [
                ],
            },
        ]
        self.image_embed_format = {"type": "image"}
        self.text_embed_format = {"type": "text", "text": ""}

    def construct_prompt(self, prompt, number_images):
        self.prompt_format["content"] = [self.image_embed_format * number_images]
        self.text_embed_format["text"] = self.image_prompt
        self.prompt_format["content"].append(self.text_embed_format)

    @timer
    def _call_model(self, inputs):
        return self.model.generate(**inputs, max_new_tokens=200, do_sample=False)

    def call_model(self, prompt, images):
        # construct prompt in place
        self.construct_prompt(prompt, len(images))
        conversation = self.processor.apply_chat_template(self.prompt_format, add_generation_prompt=True)
        inputs = self.processor(images=images, text=conversation, return_tensors="pt").to("cuda", torch.float16)

        result, time = self._call_model(inputs)
        clean = self.processor.decode(result[0][2:], skip_special_tokens=True)
        return clean, time

# 7b param
class LLaVA_One_Vision(LLaVA):
    def __init__(self):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        super().__init__(
            processor_class=AutoProcessor,
            model_class=LlavaOnevisionForConditionalGeneration,
            model_path="llava-hf/llava-onevision-qwen2-7b-ov-hf",
            image_prompt="These images are frames of a video, what do they depict? Do not structure your response by frame.",
            title_prompt="Create a fitting title of a video with this summary: ",
            summary_prompt="The text describes a video, explain in detail what happens: "
        )

# 7b param
class LLaVA_Next_Video(LLaVA):
    def __init__(self):
        from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
        super().__init__(
            processor_class=LlavaNextVideoProcessor,
            model_class=LlavaNextVideoForConditionalGeneration,
            model_path="llava-hf/LLaVA-NeXT-Video-7B-hf",
            image_prompt="These images are frames of a video, what do they depict? Do not structure your response by frame.",
            title_prompt="Create a fitting title of a video with this summary: ",
            summary_prompt="The text describes a video, explain in detail what happens: "
        )

