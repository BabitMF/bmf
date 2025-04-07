from models.hugging_face.deepseek import Deepseek_VL2, Deepseek_Janus_1b, Deepseek_Janus_7b
from models.hugging_face.qwen import Qwen2_VL, Qwen2_5_VL_3b, Qwen2_5_VL_7b
from models.hugging_face.llava import LLaVA_One_Vision, LLaVA_Next_Video

from models.vllm.deepseek_vllm import Deepseek_VL2_VLLM
from models.vllm.qwen_vllm import Qwen2_VL_VLLM, Qwen2_5_VL_3b_VLLM, Qwen2_5_VL_7b_VLLM
from models.vllm.llava_vllm import LLaVA_Next_Video_VLLM, LLaVA_One_Vision_VLLM
# only entry point exposed to llm_caption.py
# calls this to create the model class
class ModelFactory:
    MODEL_MAP= {
        "vllm": {
            "Deepseek_VL2": Deepseek_VL2_VLLM,
            # janus models are not supported in vllm
            # "Deepseek_Janus_1b": Deepseek_Janus_1b_VLLM,
            # "Deepseek_Janus_7b": Deepseek_VLLM,
            "Qwen2_VL": Qwen2_VL_VLLM,
            "Qwen2_5_VL_3b": Qwen2_5_VL_3b_VLLM,
            "Qwen2_5_VL_7b": Qwen2_5_VL_7b_VLLM,
            "LLaVA_One_Vision": LLaVA_One_Vision_VLLM,
            "LLaVA_Next_Video": LLaVA_Next_Video_VLLM
        },
        "hugging_face": {
            "Deepseek_VL2": Deepseek_VL2,
            "Deepseek_Janus_1b": Deepseek_Janus_1b,
            "Deepseek_Janus_7b": Deepseek_Janus_7b,
            "Qwen2_VL": Qwen2_VL,
            "Qwen2_5_VL_3b": Qwen2_5_VL_3b,
            "Qwen2_5_VL_7b": Qwen2_5_VL_7b,
            "LLaVA_One_Vision": LLaVA_One_Vision,
            "LLaVA_Next_Video": LLaVA_Next_Video
        }
    }

    def __init__(self, model_name, backend, batch_size):
        # check for valid backend
        if backend not in {"vllm", "hugging_face"}:
            raise ValueError(f"Unknown backend: {backend}, Valid backends: {list(self.MODEL_MAP.keys())}")
        # extract class to create
        model = self.MODEL_MAP[backend].get(model_name, None)
        # check if model exists
        if model is None:
            raise ValueError(f"Unknown model: {model_name}, Valid models: {list(self.MODEL_MAP.keys())}")
        # if vllm backend, give batch size as it needs a max multimodal input size
        if backend == 'vllm':
            self.model = model(batch_size)
        # otherwise instantiate extended hugging face class
        else:
            self.model = model()

    def get_model(self):
        return self.model
