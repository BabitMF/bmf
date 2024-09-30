import os
import torch
import torch.nn as nn
import argparse
try:
    import llava
except:
    print("Please install llava.")

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

class ModelWrapper(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
    
    def forward(self, images):
        return self.model.encode_images(images)


parser = argparse.ArgumentParser(description='export vit')
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()
if not os.path.exists(args.output):
    os.makedirs(args.output)

model_path = args.input
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
        )

model.model.eval()
print(model.model)
torch.save(model.model.image_newline, args.output + "/image_newline.pth")
image_newline_data = model.model.image_newline.cpu().detach().numpy()
image_newline_data.tofile(args.output + "/image_newline_c++.pth")

visual_model = ModelWrapper(model)
dynamic_axes = {'x': {0: 'batch_size'}}

x = torch.randn(1, 3, 336, 336, device='cuda', dtype=torch.float16)
# x = torch.randn(1, 3, 384, 384, device='cuda', dtype=torch.float16)
torch.onnx.export(visual_model, x, args.output + "/vit.onnx",
                  input_names=['x'],
                  output_names=['output'],
                  opset_version=17,
                  dynamic_axes=dynamic_axes)