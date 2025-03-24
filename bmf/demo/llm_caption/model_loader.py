from transformers import AutoModelForCausalLM
import torch
import os
from abc import ABC, abstractmethod
import time

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

# decorator to time a function call for metrics
def timer(func):
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        inference_time = time.time() - start
        return result, inference_time
    return wrapper


# base class, all models extend and define the call_model function used by llm_caption module
class ModelFactory(ABC):
    def __init__(self):
        # needed by all models
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    @abstractmethod
    def call_model(self, images):
        pass

# wrapper class for deep seek modules, since call_model is very similar
class Deepseek(ModelFactory, ABC):
    def __init__(self, processor_class, 
                 model_path,
                 model_function,
                 image_embed_format,
                 image_prompt,
                 title_prompt,
                 summary_prompt):
        super().__init__()
        self.vl_chat_processor = processor_class.from_pretrained(model_path)
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
        self.model_function = self.get_nested_attr(self.vl_gpt, model_function)
        self.image_embed_format = image_embed_format
        self.image_prompt = image_prompt
        self.title_prompt = title_prompt
        self.summary_prompt = summary_prompt
        # common prompt format
        self.prompt_format = [
            {
                "role": "<|User|>",
                "content": "",
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

    # helper for deepseek family, attr_path is either
    # "language.generate" or "language_model.generate"
    # and need to get the nested generate function (the actual model call)
    # e.g. vl_gpt.language.generate(...)
    def get_nested_attr(self, obj, attr_path):
        attributes = attr_path.split(".")
        for attr in attributes:
            obj = getattr(obj, attr)
        return obj

    # modifies the same prompt
    def construct_prompt(self, content, number_images):
        # modify the conversation prompt
        self.prompt_format[0]["content"] = self.image_embed_format * number_images + content
    
    # times the function and returns a tuple of the (function output, time)
    @timer
    def _call_model(self, inputs_embeds, prepare_inputs, tokenizer):
        # run the model to get the response using the function pointer stored during initialisation
        return self.model_function(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

    # returns tuple of (response, inference time)
    def call_model(self, prompt, images):
        """Calls the model with a conversation prompt, and a buffer"""
        # constructs a prompt
        self.construct_prompt(prompt, len(images))

        tokenizer = self.vl_chat_processor.tokenizer
        prepare_inputs = self.vl_chat_processor(
            conversations=self.prompt_format,
            images=images,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)
        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        outputs, inference_time = self._call_model(inputs_embeds, prepare_inputs, tokenizer)
        print(f"Inference time on batch with {(str(len(images)) + ' frames') if len(images) != 0 else 'title/summary'}: ", round(inference_time, 2))

        # remove the special marking at the end of the answer and return it
        return (tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True), inference_time)

# tested with python 3.8
# 3B params
class Deepseek_VL2(Deepseek):
    def __init__(self):
        """Initialises deepseek vl2 model, documentation: https://huggingface.co/deepseek-ai/deepseek-vl2-tiny"""
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
        super().__init__(
            processor_class=DeepseekVLV2Processor,
            model_path="deepseek-ai/deepseek-vl2-tiny",
            model_function="language.generate",
            image_embed_format="<image>",
            image_prompt=" These images are frames of a video, what do they depict? Do not structure your response by frame.",
            title_prompt="Create a fitting title of a video with this summary: ",
            summary_prompt="The text describes a video, explain in detail what happens: "
        )
        print("Using Deepseek_VL2")

# tested with python 3.8
# 1B params
class Deepseek_Janus(Deepseek):
    def __init__(self):
        """Initialises deepseek Janus model, documentation: https://huggingface.co/deepseek-ai/Janus-Pro-1B"""
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        super().__init__(
            processor_class=VLChatProcessor,
            model_path="deepseek-ai/Janus-Pro-1B",
            model_function="language_model.generate",
            image_embed_format="<image_placeholder>",
            image_prompt=" Do not repeat the prompt, these images are frames of a video, what do they depict? Include as much detail as possible, do not talk about in frames and structure your response with 'The video depicts'",
            title_prompt="Create a title for a video with this summary: ",
            summary_prompt="Summarise in detail what happens in this video summary: "
        )
        print("Using Deepseek_Janus")

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
