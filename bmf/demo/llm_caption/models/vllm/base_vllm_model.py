from models.base_model import BaseVisionModel
from abc import ABC, abstractmethod
from utils.timer import timer
from vllm import LLM, SamplingParams

class BaseVLLMVisionModel(BaseVisionModel, ABC):
    def __init__(self, model_name, 
                 batch_size, 
                 prompt_format, 
                 image_embed, 
                 image_prompt, 
                 title_prompt, 
                 summary_prompt,
                 overrides=None):
        super().__init__()
        self.model = LLM(model=model_name,
                         limit_mm_per_prompt={"image": batch_size},
                         max_num_seqs=1,
                         hf_overrides=overrides)
        self.sample_params = SamplingParams(max_tokens=512)
        self.prompt_template = {
            "prompt": "",
            "multi_modal_data": {
                "image": []
            }
        }
        self.prompt_format = prompt_format
        self.image_embed = image_embed
        self.image_prompt = image_prompt
        self.title_prompt = title_prompt 
        self.summary_prompt = summary_prompt 

    @timer
    def _call_model(self, prompt):
        return self.model.generate(prompt, sampling_params=self.sample_params)

    def call_model(self, prompt, images):
        self.prompt_template["multi_modal_data"]["image"] = images
        self.prompt_template["prompt"] = self.prompt_format.format(q=prompt, i=self.image_embed * len(images))

        # remove the multi_modal_data key if no images used
        if not images:
            tmp = self.prompt_template.copy()
            del tmp["multi_modal_data"]
            response, inference_time = self._call_model([tmp])
        else:
            response, inference_time = self._call_model([self.prompt_template])

        self.log_time(inference_time, len(images))
        formatted = ""
        for r in response:
            formatted += r.outputs[0].text.encode('utf-8').decode('unicode_escape')
        return formatted, inference_time


