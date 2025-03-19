import torch
from bmf import VideoFrame, Module, Timestamp, ProcessResult, Packet
import bmf.hml.hmp as mp
from PIL import Image
import numpy as np
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
import os
import json
import time
import threading

def convert_to_pil(pkt):
    vf = pkt.get(VideoFrame)
    rgb = mp.PixelInfo(mp.kPF_RGB24)
    numpy_vf = vf.reformat(rgb).frame().plane(0).numpy()
    return Image.fromarray(numpy_vf)

def read_json(file_path):
    if os.path.getsize(file_path) == 0:
        return None
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def write_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def init_model(model_name):
    if model_name == "Deepseek_VL2":
        return Deepseek_VL2()
    elif model_name == "Deepseek_Janus":
        return Deepseek_Janus()
    elif model_name == "Qwen2_VL":
        return Qwen2_VL()
    else:
        # default to janus
        return Deepseek_Janus()

class llm_caption(Module):

    def __init__(self, node, option=None):
        # initialise model
        self.init_model()
        # list of PIL images to be inferenced - cannot exceed BATCH_SIZE
        self.buffer = []
        # concatenated answer of each summary on a batch
        self.combined_answer = ""

        # how many images can be inferenced in a single prompt
        self.batch_size = 4
        if option and "batch_size" in option.keys():
            self.batch_size = option["batch_size"]
        print("Batch size: ", self.batch_size)

        # where to write the output
        self.output_path = "caption.json"
        if option and "result_path" in option.keys():
            self.output_path = option["result_path"]
        print("Writing output to: ", self.output_path)
        # wipe the file first
        with open(self.output_path, "w") as file:
            pass
        
        # whether output from previous nodes should be propagated out
        self.pass_through = False
        if option and "pass_through" in option.keys():
            self.pass_through = option["pass_through"]

        # first frame extracted is always black
        self.skip_first = True
        # metric to see how long it takes
        self.total_inferencing_time = 0

        # not multithreaded to begin with - high gpu memory needed
        self.multithreading = False
        # check if multithreading is enabled
        if option and "multithreading" in option.keys() and option["multithreading"]:
            self.multithreading = True
            # default to 2 threads
            max_threads = 2
            if option and "max_threads" in option.keys():
                max_threads = option["max_threads"]
            # only allow max_threads to be inferencing
            self.semaphore = threading.Semaphore(max_threads)

            # list of tuples (id, result, frames), is unordered
            self.thread_result = []
            # to sort in chronological order when joining results
            self.thread_id = 0
            # list of threads to join at the end
            self.threads = []
            # mutex lock
            self.lock = threading.Lock()
        print("Multithreading: ", self.multithreading)
        if self.multithreading:
            print("Max threads: ", max_threads)

    def init_model(self):
        """Initialises the model"""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model_path = "deepseek-ai/deepseek-vl2-tiny"
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
        vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    def call_model(self, conversation, buffer):
        """Calls the model with a conversation prompt, and a buffer"""
        tokenizer = self.vl_chat_processor.tokenizer
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=buffer,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        start = time.time()
        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        inference_time = time.time() - start

        # if multithreaded, lock first
        if self.multithreading:
            with self.lock:
                self.total_inferencing_time += inference_time
        else:
            self.total_inferencing_time += inference_time
        print(f"Inference time on batch with {len(buffer)} frames: ", round(inference_time, 2))

        # remove the special marking at the end of the answer and return it
        return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)[:-len("<｜end▁of▁sentence｜>")]

    def log_result(self, answer):
        """Is only called when not multithreaded, each call to the model is logged to the disk directly"""
        # combine answer for summary and title at the end
        self.combined_answer += answer
        data = read_json(self.output_path)
        # if no data, create the schema first
        if data is None:
            data = {
                "video_title": "",
                "batch_size": self.batch_size,
                "batches": [],
                "frames_analysed" : 0,
                "summary": "",
            }

        # append the new data
        new_batch = {
            "batch_id": len(data["batches"]) + 1,
            "frames": len(self.buffer),
            "description": answer,
        }
        data["batches"].append(new_batch)
        data["frames_analysed"] += len(self.buffer)

        # write to disk immediately
        write_json(data, self.output_path)

    def prepare_call(self, id, buffer):
        """Creates and embeds the images into the prompt. buffer is either self.buffer or a thread's own buffer"""
        number_images = len(buffer)
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\n" * number_images + " These images are frames of a video, what do they depict? Do not structure your response by frame.",
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

        # get the reponse of the inference
        answer = self.call_model(conversation, buffer)

        # if its multithreaded, release semaphore count to allow another thread
        # then append the result to result list
        if self.multithreading:
            self.semaphore.release()
            with self.lock:
                self.thread_result.append((id, answer, number_images))
        # otherwise write the result directly to disk (append)
        else:
            self.log_result(answer)

    def join_and_sort_results(self):
        """Only called when multithreaded, joins threads, bucket sort results in chronological order and prepares result to be written to disk"""
        # join all threads together (blocking)
        for thread in self.threads:
            thread.join()

        # create the schema
        data = {
            "video_title": "",
            "batch_size": self.batch_size,
            "batches": [{"batch_id": 0,
                        "frames": 0,
                        "description": ""} for _ in range(len(self.threads))],
            "frames_analysed" : 0,
            "summary": "",
        }

        answer_buckets = [""] * len(self.threads)
        for id, result, number in self.thread_result:
            # update number of frames analysed
            data["frames_analysed"] += number
            # select correct bucket to write result
            curr = data["batches"][id]
            # thread_id starts at 0, batch starts at 1
            # fill in results into the bucket
            curr["batch_id"] = id + 1
            curr["frames"] = number
            curr["description"] = result

            # sort the result of each batch into bucket as well
            answer_buckets[id] = result

        # combined answer is now chronologically concatenated
        for tmp in answer_buckets:
            self.combined_answer += tmp

        # writes results to file, no title or summary
        write_json(data, self.output_path)

    def write_title_and_summary(self):
        """Uses the combined answer to create a summary, and also a title from that"""

        data = read_json(self.output_path)
        # get a cohesive summary
        conversation = [
            {
                "role": "<|User|>",
                "content": f"The text describes a video, explain in detail what happens: {self.combined_answer}",
            },
            {"role": "<|Assistant|>", "content": ""}
        ]
        data["summary"] = self.call_model(conversation, [])

        # get a corresponding title
        conversation = [
            {
                "role": "<|User|>",
                "content": f"Create a fitting title of a video with this summary {self.combined_answer}"
            },
            {"role": "<|Assistant|>", "content": ""}
        ]
        data["video_title"] = self.call_model(conversation, [])

        # write the updated json to the output
        write_json(data, self.output_path)
        print("Average inference time per frame: ", round(self.total_inferencing_time / data["frames_analysed"], 4))
        return

    def spawn_thread_and_reset(self):
        """Spawn a thread if multithreaded, then calls model directly"""
        # if multithreaded, creates a thread to call the model
        # and if max threads reached, blocks until a thread finishes
        if self.multithreading:
            thread = threading.Thread(target=self.prepare_call, args=(self.thread_id, self.buffer.copy()))
            self.thread_id += 1
            self.semaphore.acquire()
            self.threads.append(thread)
            thread.start()
        # otherwise does a blocking call to the model and output is written to output
        else:
            # blocking call
            self.prepare_call(0, self.buffer)

        # clear the buffer for new frames
        self.buffer = []

    def clean_up(self):
        """Cleans up all the resources and writes the title and summary"""
        if self.multithreading:
            self.join_and_sort_results()
        self.write_title_and_summary()

    def process(self, task):
        """Handles main event loop which contains list of extracted frames at specified fps"""
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]
        while not input_queue.empty():
            pkt = input_queue.get()

            # if last packet, clean up
            if pkt.timestamp == Timestamp.EOF:
                # send last batch
                if self.buffer:
                    self.spawn_thread_and_reset()
                self.clean_up()
                task.set_timestamp(Timestamp.DONE)

                for key in task.get_outputs():
                    task.get_outputs()[key].put(Packet.generate_eof_packet())
                break

            # skip first frame as its black
            if self.skip_first:
                self.skip_first = False
                continue

            # converts to PIL image, for direct inference to model
            if pkt.is_(VideoFrame):
                if self.pass_through:
                    output_queue.put(pkt)
                pil_image = convert_to_pil(pkt)
                self.buffer.append(pil_image)
                # batch size reached, send to model
                if len(self.buffer) == self.batch_size:
                    self.spawn_thread_and_reset()
        # normal termination
        return ProcessResult.OK
