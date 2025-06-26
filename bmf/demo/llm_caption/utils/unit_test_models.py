import sys
import os
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir) 

from models.model_loader import init_model

input_path = "../../../../big_bunny_1min_30fps.mp4"
combined_answer = "The video depicts a serene and peaceful landscape, with a large, ancient tree at the center of the scene. The tree's branches stretch out towards the sky, creating a canopy of leaves that filters the sunlight. The ground beneath the tree is covered in a thick layer of moss, and a small stream runs through the area, its gentle flow adding to the tranquility of the scene. The overall atmosphere is one of calmness and serenity, inviting viewers to immerse themselves in the natural beauty of the landscape."
combined_answer1 = "The video depicts a serene and peaceful landscape, with a blend of natural elements such as trees, bushes, and hills. The sky is painted in soft pastel colors, creating a calming and dreamy atmosphere. The overall composition evokes a sense of peace and natural beauty, making it an ideal setting for a calming and reflective experience."

MODELS = ["Deepseek_VL2", "Deepseek_Janus_1b", "Deepseek_Janus_7b", "Qwen2_VL", "Qwen2_5_VL_3b", "Qwen2_5_VL_7b", "LLaVA_One_Vision", "LLaVA_Next_Video"]

model = init_model(MODELS[1])
result = model.call_model("Respond with only a short title, do not repeat the description, do not label the title, do not respond with a sentence. Create a creative and fitting title for a video with a description: " + combined_answer1, [])
print(result)
