from models.deepseek import Deepseek_VL2, Deepseek_Janus
from models.qwen import Qwen2_VL, Qwen2_5_VL

# only entry point exposed to llm_caption.py
# calls this to create the model class
def init_model(model_name):
    if model_name == "Deepseek_VL2":
        return Deepseek_VL2()
    elif model_name == "Deepseek_Janus":
        return Deepseek_Janus()
    elif model_name == "Qwen2_VL":
        return Qwen2_VL()
    elif model_name == "Qwen2_5_VL":
        return Qwen2_5_VL()
    else:
        # default to janus
        return Deepseek_Janus()

