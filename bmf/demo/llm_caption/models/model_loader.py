from models.deepseek import Deepseek_VL2, Deepseek_Janus_3b, Deepseek_Janus_7b
from models.qwen import Qwen2_VL, Qwen2_5_VL_3b, Qwen2_5_VL_7b

# only entry point exposed to llm_caption.py
# calls this to create the model class
def init_model(model_name):
    if model_name == "Deepseek_VL2":
        return Deepseek_VL2()
    elif model_name == "Deepseek_Janus_3b":
        return Deepseek_Janus_3b()
    elif model_name == "Deepseek_Janus_7b":
        return Deepseek_Janus_7b()
    elif model_name == "Qwen2_VL":
        return Qwen2_VL()
    elif model_name == "Qwen2_5_VL_3b":
        return Qwen2_5_VL_3b()
    elif model_name == "Qwen2_5_VL_7b":
        return Qwen2_5_VL_7b()
    else:
        # default to janus
        return Deepseek_Janus()

