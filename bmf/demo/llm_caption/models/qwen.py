from models.base_model import ModelFactory
from abc import ABC, abstractmethod
from utils.timer import timer
# requires python >= 3.9 and newest beta release of transformers - follow documentation in docstrings
# wrapper class for qwen family
class Qwen(ModelFactory, ABC):
    def __init__(self, model_class, model_path):
        super().__init__()
        from qwen_vl_utils import process_vision_info
        from transformers import AutoTokenizer, AutoProcessor
        # load model from class import
        self.model = model_class.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
        # load auto processor from model path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.vision = process_vision_info
        self.prompt_format = [
            {
                "role": "user",
                "content": [
                ],
            }
        ]
        self.image_embed_format = {"type": "image", "image": ""}
        self.text_embed_format = {"type": "text", "text": ""}
        self.image_prompt = " Do not repeat the prompt, these images are frames of a video, what do they depict? Include as much detail as possible, do not talk about in frames and structure your response with 'The video depicts'"
        self.title_prompt = "Create a title for a video with this summary: "
        self.summary_prompt = "Summarise in detail what happens in this video summary: "

    def construct_prompt(self, prompt, images):
        ls = self.prompt_format[0]["content"]
        ls.clear()
        # image embed format needs to be copied each time and changed
        for image in images:
            tmp = self.image_embed_format.copy()
            tmp["image"] = image
            ls.append(tmp)
        # text embed format is re used everytime
        self.text_embed_format["text"] = prompt
        ls.append(self.text_embed_format)

    @timer
    def _call_model(self, inputs):
        return self.model.generate(**inputs, max_new_tokens=128)

    def call_model(self, prompt, images):
        # changes prompt in place
        self.construct_prompt(prompt, images)

        text = self.processor.apply_chat_template(
            self.prompt_format, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.vision(self.prompt_format)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids, inference_time = self._call_model(inputs)
        print(f"Inference time on batch with {(str(len(images)) + ' frames') if len(images) != 0 else 'title/summary'}: ", round(inference_time, 2))

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0], inference_time

# 2b parms, qwen2 model
class Qwen2_VL(Qwen):
    def __init__(self):
        """Initialises Qwen2 VL model, documentation: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct"""
        from transformers import Qwen2VLForConditionalGeneration
        super().__init__(
            model_class=Qwen2VLForConditionalGeneration,
            model_path="Qwen/Qwen2-VL-2B-Instruct"
        )
        print("Using Qwen2_VL")

# 3b params, qwen2.5 model
class Qwen2_5_VL(Qwen):
    def __init__(self):
        """Initialises Qwen2.5 VL model, documentation: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"""
        from transformers import Qwen2_5_VLForConditionalGeneration
        super().__init__(
            model_class=Qwen2_5_VLForConditionalGeneration,
            model_path="Qwen/Qwen2.5-VL-3B-Instruct"
        )
        print("Using Qwen2_5_VL")
