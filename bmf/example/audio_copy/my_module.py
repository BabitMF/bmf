from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame


class my_module(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.option_ = option
        pass

    def process(self, task):
        if 'exception' in self.option_:
            raise Exception("python module raise an exception during process")
        for (input_id, input_packets) in task.get_inputs().items():

            # output queue
            output_packets = task.get_outputs()[input_id]

            while not input_packets.empty():
                pkt = input_packets.get()

                # process EOS
                if pkt.timestamp == Timestamp.EOF:
                    Log.log_node(LogLevel.DEBUG, task.get_node(), "Receive EOF")
                    output_packets.put(Packet.generate_eof_packet())
                    task.timestamp = Timestamp.DONE
                    return ProcessResult.OK

                # copy input packet to output
                if pkt.defined() and pkt.timestamp != Timestamp.UNSET:
                    audio_frame = pkt.get(AudioFrame)
                    new_audio_frame = audio_frame.clone()

                    output_packet = Packet(new_audio_frame)
                    output_packet.timestamp = pkt.timestamp
                    output_packets.put(output_packet)

        # exit(0)
        return ProcessResult.OK
