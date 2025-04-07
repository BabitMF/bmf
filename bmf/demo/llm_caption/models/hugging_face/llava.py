from models.base_model import BaseVisionModel
from abc import ABC, abstractmethod
from utils.timer import timer
import torch
# tested with python3.10 and transformers >= 4.45.0 for one vision, >= 4.42.0 for next video
class LLaVA(BaseVisionModel, ABC):
    def __init__(self, processor_class,
                 model_class, 
                 model_path,
                 image_prompt,
                 title_prompt,
                 summary_prompt,
                 clean_prompt):
        super().__init__()
        # requries flash attention 2 installed and pip install bitsandbytes and pip install protobuf
        self.model = model_class.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 torch_dtype="auto", 
                                                 device_map="auto", 
                                                 attn_implementation="flash_attention_2").to("cuda")
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
        self.image_prompt = image_prompt
        self.title_prompt = title_prompt
        self.summary_prompt = summary_prompt
        self.clean_prompt = clean_prompt

    def construct_prompt(self, prompt, number_images):
        self.prompt_format[0]["content"] = [self.image_embed_format for _ in range(number_images)]
        self.text_embed_format["text"] = prompt
        self.prompt_format[0]["content"].append(self.text_embed_format)

    @timer
    def _call_model(self, inputs):
        return self.model.generate(**inputs, max_new_tokens=512, do_sample=False)

    def call_model(self, prompt, images):
        # construct prompt in place
        self.construct_prompt(prompt, len(images))
        conversation = self.processor.apply_chat_template(self.prompt_format, add_generation_prompt=True)
        inputs = self.processor(images=images if images else None, text=conversation, return_tensors="pt").to("cuda", torch.float16)

        result, time = self._call_model(inputs)
        self.log_time(time, len(images))
        clean = self.processor.decode(result[0][2:], skip_special_tokens=True)[self.clean_prompt(prompt, len(images)):]
        return clean, time

# 7b param requires transformers >= 4.45.0
class LLaVA_One_Vision(LLaVA):
    def __init__(self):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        super().__init__(
            processor_class=AutoProcessor,
            model_class=LlavaOnevisionForConditionalGeneration,
            model_path="llava-hf/llava-onevision-qwen2-7b-ov-hf",
            image_prompt="These images are frames of a video, what do they depict? Do not structure your response by frame. Begin with 'The video'",
            title_prompt="Create a short title of a video with this summary: ",
            summary_prompt="The text describes a video, explain in detail what happens: ",
            clean_prompt=lambda prompt, _: len(prompt+"assistant\n\n ") 
        )

# 7b param requires transformers >= 4.42.0
class LLaVA_Next_Video(LLaVA):
    def __init__(self):
        from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
        super().__init__(
            processor_class=LlavaNextVideoProcessor,
            model_class=LlavaNextVideoForConditionalGeneration,
            model_path="llava-hf/LLaVA-NeXT-Video-7B-hf",
            image_prompt="These images are frames of a video, what do they depict? Begin with 'The video'",
            title_prompt="Create a short title of a video with this summary: ",
            summary_prompt="The text describes a video, explain in detail what happens: ",
            clean_prompt=lambda prompt, number_images: len(prompt + "ER: ASSISTANT : ") + number_images
        )

