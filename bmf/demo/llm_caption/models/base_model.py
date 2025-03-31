import os
from abc import ABC, abstractmethod
# base class, all models extend and define the call_model function used by llm_caption module
class BaseVisionModel(ABC):
    def __init__(self):
        # needed by all models
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.zero_shot_cot_prompt = "{q} Let's think step by step"
    
    @abstractmethod
    def call_model(self, images):
        pass

    def log_time(self, time, number_frames):
        print(f"Inference time on batch with {(str(number_frames) + ' frames') if number_frames != 0 else 'title/summary'}: ", round(time, 2))

