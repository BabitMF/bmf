from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType


class pass_through(Module):

    def __init__(self, node, option=None):
        self.node_ = node

    def reset(self):
        Log.log_node(LogLevel.DEBUG, self.node_, " is doing reset")

    def process(self, task):
        for (input_id, input_packets) in task.get_inputs().items():
            # output queue
            output_packets = task.get_outputs()[input_id]
            while not input_packets.empty():
                pkt = input_packets.get()
                if pkt.get_timestamp() == Timestamp.EOF:
                    Log.log_node(LogLevel.DEBUG, task.get_node(),
                                 "Receive EOF")
                    output_packets.put(Packet.generate_eof_packet())
                    task.set_timestamp(Timestamp.DONE)
                elif pkt.get_timestamp() != Timestamp.UNSET:
                    Log.log_node(LogLevel.DEBUG, task.get_node(),
                                 ' timestamp of this processing packet is: ',
                                 pkt.get_timestamp())
                    output_packets.put(pkt)
        return ProcessResult.OK
