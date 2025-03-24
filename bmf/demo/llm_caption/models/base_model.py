import os
from abc import ABC, abstractmethod
# base class, all models extend and define the call_model function used by llm_caption module
class ModelFactory(ABC):
    def __init__(self):
        # needed by all models
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    @abstractmethod
    def call_model(self, images):
        pass
