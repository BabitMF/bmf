from models.base_model import ModelFactory
from abc import ABC, abstractmethod
from utils.timer import timer
from transformers import AutoModelForCausalLM
import torch
class Deepseek(ModelFactory, ABC):
    def __init__(self, processor_class, 
                 model_path,
                 model_function,
                 image_embed_format,
                 image_prompt,
                 title_prompt,
                 summary_prompt):
        super().__init__()
        self.vl_chat_processor = processor_class.from_pretrained(model_path,
                                                 torch_dtype="auto", 
                                                 device_map="auto", 
                                                 attn_implementation="flash_attention_2")
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

# 1B params
class Deepseek_Janus_3b(Deepseek):
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
        print("Using Deepseek_Janus_3b")

# 7B params
class Deepseek_Janus_7b(Deepseek):
    def __init__(self):
        """Initialises deepseek Janus model, documentation: https://huggingface.co/deepseek-ai/Janus-Pro-1B"""
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        super().__init__(
            processor_class=VLChatProcessor,
            model_path="deepseek-ai/Janus-Pro-7B",
            model_function="language_model.generate",
            image_embed_format="<image_placeholder>",
            image_prompt=" Do not repeat the prompt, these images are frames of a video, what do they depict? Include as much detail as possible, do not talk about in frames and structure your response with 'The video depicts'",
            title_prompt="Create a title for a video with this summary: ",
            summary_prompt="Summarise in detail what happens in this video summary: "
        )
        print("Using Deepseek_Janus_7b")
