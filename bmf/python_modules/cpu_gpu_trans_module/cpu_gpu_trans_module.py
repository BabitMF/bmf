import bmf
from bmf import *


class cpu_gpu_trans_module(Module):

    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        self.eof_received_ = False
        if 'to_gpu' in option.keys():
            self.trans_to_gpu_ = option['to_gpu']
        self.gpu_alloced_ = []
        self.init_context_flag_ = False
        self.ctx = None

    def process(self, task):

        # get input and output packet queue
        input_queue = task.get_inputs()[0]
        output_queue = task.get_outputs()[0]

        # add all input frames into frame cache
        while not input_queue.empty():
            in_pkt = input_queue.get()

            if in_pkt.get_timestamp() == Timestamp.EOF:
                # we should done all frames processing in following loop
                self.eof_received_ = True
                continue

            if self.trans_to_gpu_ == 1:
                in_frame = in_pkt.get(bmf.VideoFrame)
                gpu_frame = in_frame.cuda()
                gpu_frame.pts = in_frame.pts
                gpu_frame.time_base = in_frame.time_base
                out_pkt = Packet(gpu_frame)
            else:
                in_frame = in_pkt.get(bmf.VideoFrame)
                video_frame = in_frame.cpu()
                video_frame.pts = in_frame.pts
                video_frame.time_base = in_frame.time_base
                out_pkt = Packet(video_frame)

            out_pkt.set_timestamp(in_pkt.get_timestamp())
            output_queue.put(out_pkt)

        # add eof packet to output
        if self.eof_received_:
            output_queue.put(Packet.generate_eof_packet())
            Log.log_node(LogLevel.DEBUG, self.node_, 'output stream', 'done')
            task.set_timestamp(Timestamp.DONE)
        return ProcessResult.OK


def register_cpu_gpu_trans_module_info(info):
    info.module_description = "Builtin cpu/gpu conversion module"
    info.module_tag = ModuleTag.TAG_UTILS
