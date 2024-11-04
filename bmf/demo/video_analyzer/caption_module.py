import torch

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

from bmf import Module, Log, Timestamp, ProcessResult, LogLevel, Packet, VideoFrame

def qwen2_setup():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
    return model, processor

class CaptionModule(Module):
    def __init__(self, node=None, option=None):
        self._node = node
        if not option:
            Log.log_node(LogLevel.ERROR, self._node, "no option")
            return
        self.model, self.processor = qwen2_setup()
        self.batch_size = option.get("batch_size")

    def process(self, task):
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        batch = []
        while not input_queue.empty():
            in_pkt = input_queue.get()

            if in_pkt.timestamp == Timestamp.EOF:
                self.eof_received = True
                continue

            video_frame = in_pkt.get(VideoFrame)
            batch.append(video_frame)

            if len(batch) == self.batch_size:
                pass