from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame


class python_module(Module):

    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        pass

    def process(self, task):
        for (input_id, input_packets) in task.get_inputs().items():
            while not input_packets.empty():
                pkt = input_packets.get()
                # process EOS
                if pkt.timestamp == Timestamp.EOF:
                    Log.log_node(LogLevel.DEBUG, task.get_node(),
                                 "Receive EOF")
                    for output_id in task.get_outputs().keys():
                        output_packets = task.get_outputs()[output_id]
                        output_packets.put(Packet.generate_eof_packet())
                    task.timestamp = Timestamp.DONE
                    return ProcessResult.OK

                if pkt.is_(dict):
                    json = pkt.get(dict)
                    print("Python module parsed json:", json)
                for output_id in task.get_outputs().keys():
                    output_packets = task.get_outputs()[output_id]
                    output_packets.put(pkt)

        return ProcessResult.OK
