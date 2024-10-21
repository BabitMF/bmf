import bmf
import numpy as np
from bmf import ProcessResult, Packet, Timestamp, VideoFrame
from PIL import Image
import bmf.hmp as mp
from bmf.lib._bmf.sdk import ffmpeg

from diffusers import StableDiffusionUpscalePipeline
import torch
import math

debug = False
use_tile = True

def tiled_process(upscaler, prompt, np_rgb, scale_factor=4, tile_num=None, padding=0, seed=2023):
    if len(np_rgb.shape) != 3:
        print(f'tiled_process invalid input dimension: {np_rgb.shape}')

    upscaled_image_np = []
    upscaled_blk = []

    if tile_num and len(tile_num) == 2:
        assert(tile_num[0] > 0)
        assert(tile_num[1] > 0)

        new_shape = (np_rgb.shape[0]*scale_factor, np_rgb.shape[1]*scale_factor, np_rgb.shape[2])
        upscaled_image_np = np.zeros(new_shape)
        print(upscaled_image_np.shape)

        min_dim = 64
        r_s = math.ceil(np_rgb.shape[0]/tile_num[0]/min_dim) * min_dim
        c_s = math.ceil(np_rgb.shape[1]/tile_num[1]/min_dim) * min_dim

        print(f'r_s = {r_s}, c_s = {c_s}')

        for iy in range(tile_num[0]):
            for ix in range(tile_num[1]):
                print(f'iy = {iy}, ix = {ix}')
                src_start_y = max(iy * r_s - padding, 0)
                src_end_y = min((iy+1) * r_s + padding, np_rgb.shape[0])
                src_start_x = max(ix * c_s - padding, 0)
                src_end_x = min((1+ix) * c_s + padding, np_rgb.shape[1])

                print(f'src_start_y = {src_start_y}, src_end_y = {src_end_y}, src_start_x = {src_start_x}, src_end_x = {src_end_x}')

                blk = np_rgb[src_start_y:src_end_y, src_start_x:src_end_x, :]

                upscaled_blk = upscaler(prompt=prompt, generator=torch.manual_seed(seed), image=[blk], output_type='np').images[0]

                print(f'blk.shape = {blk.shape}, upscaled_blk.shape = {upscaled_blk.shape}')

                if debug:
                    output_filename = f'test/upscaled_blk_{iy}_{ix}.png'
                    temp = upscaled_blk
                    upscaled_image = np.clip(temp * 255.0, 0, 255).astype("uint8")
                    upscaled_image_pil = Image.fromarray(upscaled_image)
                    upscaled_image_pil.save(output_filename)

                blk_start_x = ix * c_s - src_start_x
                blk_end_x = min(blk_start_x + c_s, blk.shape[1])
                blk_start_y = iy * r_s - src_start_y
                blk_end_y = min(blk_start_y + r_s, blk.shape[0])

                blk_start_x = scale_factor * blk_start_x
                blk_end_x = scale_factor * blk_end_x
                blk_start_y = scale_factor * blk_start_y
                blk_end_y = scale_factor * blk_end_y

                print(f'blk_start_y = {blk_start_y}, blk_end_y = {blk_end_y}, blk_start_x = {blk_start_x}, blk_end_x = {blk_end_x}')

                dst_start_x = scale_factor * ix * c_s
                dst_end_x = min(scale_factor * (ix+1) * c_s, upscaled_image_np.shape[1])
                dst_start_y = scale_factor * iy * r_s
                dst_end_y = min(scale_factor * (iy+1) * r_s, upscaled_image_np.shape[0])

                print(f'dst_start_y = {dst_start_y}, dst_end_y = {dst_end_y}, dst_start_x = {dst_start_x}, dst_end_x = {dst_end_x}')

                upscaled_image_np[dst_start_y:dst_end_y, dst_start_x:dst_end_x,:] = upscaled_blk[blk_start_y:blk_end_y, blk_start_x:blk_end_x, :]

                if debug:
                    output_filename = f'test/tiled_{iy}_{ix}.png'
                    temp = upscaled_image_np[dst_start_y:dst_end_y, dst_start_x:dst_end_x,:]
                    upscaled_image = np.clip(temp * 255.0, 0, 255).astype("uint8")
                    upscaled_image_pil = Image.fromarray(upscaled_image)
                    upscaled_image_pil.save(output_filename)

                print(f'upscaled_image_np.shape = {upscaled_image_np.shape}, upscaled_blk.shape = {upscaled_blk.shape}')

    else:
        upscaled_image_np = upscaler(prompt=prompt, generator=torch.manual_seed(seed), image=[np_rgb], output_type='np').images[0]

    return upscaled_image_np


class py_sdsr4_module(bmf.Module):
    def __init__(self, node, option=None):
        compute_device = 'cuda'
        if not torch.cuda.is_available():
            print('warning: GPU is not available, the inference cannot work...')
            compute_device = 'cpu'
            return

        print(f'compute_device = {compute_device}')
        # load model and scheduler
        model_path = "stabilityai/stable-diffusion-x4-upscaler"
        if option and 'model_path' in option.keys():
            model_path = option['model_path']

        print(f'model_path = {model_path}')

        _upscaler = StableDiffusionUpscalePipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        self._upscaler = _upscaler.to(compute_device)
        self._prompt = ""
        self.idx = 0

        print(f'py_sdsr4_module init successfully...')

    def process(self, task):
        idx = self.idx

        for (input_id, input_queue) in task.get_inputs().items():
            # get output queue
            output_queue = task.get_outputs()[input_id]

            while not input_queue.empty():
                # get the earliest packet from queue
                packet = input_queue.get()

                # handle EOF
                if packet.timestamp == Timestamp.EOF:
                    output_queue.put(Packet.generate_eof_packet())
                    task.timestamp = Timestamp.DONE

                # process packet if not empty
                if packet.timestamp != Timestamp.UNSET and packet.is_(VideoFrame):
                    vf = packet.get(VideoFrame)
                    rgb = mp.PixelInfo(mp.kPF_RGB24)

                    np_vf = ffmpeg.reformat(vf, "rgb24").frame().plane(0).numpy()

                    print(np_vf.shape)

                    image = np_vf.astype(np.float32) / 255.0

                    seed = 2023

                    if use_tile:
                        tile_width = 192
                        tile_y_num = math.ceil(image.shape[0]/tile_width)
                        tile_x_num = math.ceil(image.shape[1]/tile_width)
                        padding = 32
                        print(f'tile_y_num = {tile_y_num}, tile_x_num = {tile_x_num}, padding = {padding}')
                        upscaled_image_np = tiled_process(self._upscaler, prompt=self._prompt, np_rgb=image, scale_factor=4, tile_num=(tile_y_num,tile_x_num), padding=padding, seed=seed)
                    else:
                        upscaled_image_np = self._upscaler(prompt=self._prompt, generator=torch.manual_seed(seed), image=[image], output_type='np').images[0]

                    upscaled_image = np.clip(upscaled_image_np * 255.0, 0, 255).astype("uint8")

                    if debug:
                        upscaled_image_pil = Image.fromarray(upscaled_image)
                        output_name = f'test/out_frame_{idx}.png'
                        print(f'output_name = {output_name}')
                        upscaled_image_pil.save(output_name)

                    self.idx = idx + 1
                    out_frame_np = np.array(upscaled_image)
                    rgb = mp.PixelInfo(mp.kPF_RGB24)
                    frame = mp.Frame(mp.from_numpy(out_frame_np), rgb)

                    out_frame = VideoFrame(frame)
                    out_frame.pts = vf.pts
                    out_frame.time_base = vf.time_base

                    pkt = Packet(out_frame)
                    pkt.timestamp = out_frame.pts

                    output_queue.put(pkt)

        return ProcessResult.OK
