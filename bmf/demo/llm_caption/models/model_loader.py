from models.deepseek import Deepseek_VL2, Deepseek_Janus_3b, Deepseek_Janus_7b
from models.qwen import Qwen2_VL, Qwen2_5_VL_3b, Qwen2_5_VL_7b
from models.llava import LLaVA_One_Vision, LLaVA_Next_Video

MODEL_MAP = {
    "Deepseek_VL2": Deepseek_VL2,
    "Deepseek_Janus_3b": Deepseek_Janus_3b,
    "Deepseek_Janus_7b": Deepseek_Janus_7b,
    "Qwen2_VL": Qwen2_VL,
    "Qwen2_5_VL_3b": Qwen2_5_VL_3b,
    "Qwen2_5_VL_7b": Qwen2_5_VL_7b,
    "LLaVA_One_Vision": LLaVA_One_Vision,
    "LLaVA_Next_Video": LLaVA_Next_Video
}
# only entry point exposed to llm_caption.py
# calls this to create the model class
def init_model(model_name):
    model = MODEL_MAP.get(model_name, None)
    if model is None:
        raise ValueError(f"Unknown model: {model_name}")
    return model()

