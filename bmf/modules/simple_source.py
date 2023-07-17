from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType


class simple_source(Module):

    def __init__(self, node, option=None):
        self.node_ = node
        self.global_timestamp_ = 1

    def reset(self):
        Log.log_node(LogLevel.DEBUG, self.node_, " is doing reset")

    def process(self, task):
        # every process call will create a packet
        for (output_id, output_queue) in task.get_outputs().items():
            if self.global_timestamp_ >= 1500:
                output_queue.put(Packet.generate_eos_packet())
                task.set_timestamp(Timestamp.DONE)
                task.get_node().close()
            elif self.global_timestamp_ % 10 == 0:
                output_queue.put(Packet.generate_eof_packet())
                task.set_timestamp(Timestamp.DONE)
            else:
                pkt = Packet()
                pkt.set_timestamp(self.global_timestamp_)
                output_queue.put(pkt)
            Log.log_node(LogLevel.DEBUG, task.get_node(),
                         "simple_source generate data time:",
                         output_queue.queue[0].get_timestamp())
            self.global_timestamp_ += 1
            return ProcessResult.OK

        # EOS
        return ProcessResult.STOP
