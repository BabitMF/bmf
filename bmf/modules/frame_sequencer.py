import sys
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType

if sys.version_info.major == 2:
    from Queue import *
else:
    from queue import *


class frame_sequencer(Module):

    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option

        self.input_inited_ = {}
        self.input_done_ = {}
        self.input_cache_queue_ = {}

        self.max_packets_for_one_loop_ = 2

    def process(self, task=None):
        if len(task.get_inputs()) != len(task.get_outputs()):
            Log.log_node(LogLevel.ERROR, self.node_,
                         'Input count not match output count')
            return ProcessResult.STOP

        # process input packets
        for (input_idx, pkt_queue) in task.get_inputs().items():
            # for every input queue, need send the first packet to downstream
            if input_idx not in self.input_inited_:
                if not pkt_queue.empty():
                    pkt = pkt_queue.get()
                    if pkt.get_timestamp() != Timestamp.UNSET:
                        task.get_outputs()[input_idx].put(pkt)
                        self.input_inited_[input_idx] = 1
                        Log.log_node(LogLevel.DEBUG, self.node_, 'init',
                                     input_idx)
            else:
                if input_idx not in self.input_cache_queue_:
                    self.input_cache_queue_[input_idx] = Queue()
                while not pkt_queue.empty():
                    pkt = pkt_queue.get()
                    if pkt.get_data() is not None or pkt.get_timestamp(
                    ) == Timestamp.EOF:
                        self.input_cache_queue_[input_idx].put(pkt)
                        Log.log_node(LogLevel.DEBUG, self.node_, 'receive',
                                     pkt.get_data(), 'time',
                                     pkt.get_timestamp(), 'from', input_idx)

        # process output packets
        for (output_idx, output_queue) in task.get_outputs().items():
            # if this input is done, move to next
            if output_idx in self.input_done_:
                continue

            if output_idx not in self.input_cache_queue_:
                return ProcessResult.OK

            processed_cnt = 0
            cache_queue = self.input_cache_queue_[output_idx]
            while processed_cnt < self.max_packets_for_one_loop_ \
                    and not cache_queue.empty():
                pkt = cache_queue.get()

                if pkt.get_timestamp() == Timestamp.EOF:
                    self.input_done_[output_idx] = 1
                    Log.log_node(LogLevel.DEBUG, self.node_, 'sent eof to',
                                 output_idx)
                    output_queue.put(pkt)
                    processed_cnt += 1
                    break

                if pkt.get_data() is not None:
                    output_queue.put(pkt)
                    Log.log_node(LogLevel.DEBUG, self.node_, 'send',
                                 pkt.get_data(), 'to', output_idx)
                    processed_cnt += 1

            return ProcessResult.OK

        # if all output done, node is done
        if len(self.input_done_) == len(task.get_outputs()):
            Log.log_node(LogLevel.DEBUG, self.node_, 'processing done')
            task.set_timestamp(Timestamp.DONE)

        return ProcessResult.OK
