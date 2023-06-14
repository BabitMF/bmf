
import bmf
import numpy as np
from bmf import ProcessResult, Packet, Timestamp, VideoFrame
import PIL
import bmf.hml.hmp as mp

from deoldify import device
from deoldify.device_id import DeviceId
import torch
from deoldify.visualize import *
import warnings

debug = False

class py_deoldify_module(bmf.Module):
    def __init__(self, node, option=None):
        print(f'py_deoldify_module init ...')
        self.node_ = node
        self.option_ = option
        print(option)
        warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
        
        #NOTE:  This must be the first call in order to work properly!
        #choices:  CPU, GPU0...GPU7
        device.set(device=DeviceId.GPU0)

        if not torch.cuda.is_available():
            print('warning: GPU is not available, the computation is going to be very slow...')

        weight_path=Path('/content/DeOldify')
        if option and 'model_path' in option.keys():
            model_path = option['model_path']
            if not model_path:
                print(f'model_path={model_path}')
                weight_path=Path(model_path)

        self.colorizer = get_stable_video_colorizer(weight_path)
        self.idx = 0

        print(f'py_deoldify_module init successfully...')


    def process(self, task):
        # iterate through all input queues to the module
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
                    np_vf = vf.reformat(rgb).frame().plane(0).numpy()

                    # numpy to PIL
                    image = Image.fromarray(np_vf.astype('uint8'), 'RGB')

                    colored_image = self.colorizer.colorize_single_frame_from_image(image)

                    if not colored_image:
                        print(f'Fail to process the input image with idx = {idx}')
                        continue

                    if debug:
                        input_name = f'video/bmf_raw/frame_{idx}.png'
                        print(f'input_name = {input_name}')
                        image.save(input_name)

                        output_name = f'video/bmf_out/frame_{idx}.png'
                        print(f'output_name = {output_name}')
                        colored_image.save(output_name)

                    self.idx = idx + 1
                    out_frame_np = np.array(colored_image)
                    rgb = mp.PixelInfo(mp.kPF_RGB24)
                    frame = mp.Frame(mp.from_numpy(out_frame_np), rgb)

                    out_frame = VideoFrame(frame)
                    out_frame.pts = vf.pts
                    out_frame.time_base = vf.time_base

                    pkt = Packet(out_frame)
                    pkt.timestamp = out_frame.pts

                    output_queue.put(pkt)


        return ProcessResult.OK
