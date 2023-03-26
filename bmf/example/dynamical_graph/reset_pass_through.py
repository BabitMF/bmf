from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType


class reset_pass_through(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.eof_ = False

    def reset(self):
        Log.log_node(LogLevel.DEBUG, self.node_, " is doing reset")

    def close(self):
        return 0

    def dynamic_reset(self, opt_reset=None):
        print("---Dynamical reset the option---")
        print(opt_reset)
        print("--------------------------------")


    def process(self, task):
        for (input_id, input_packets) in task.get_inputs().items():
            # output queue
            output_packets = None
            if len(task.get_outputs()) > 0 and input_id < len(task.get_outputs()):
                output_packets = task.get_outputs()[input_id]
            while not input_packets.empty():
                pkt = input_packets.get()
                if pkt.get_timestamp() == Timestamp.EOF:
                    Log.log_node(LogLevel.INFO, task.get_node(), "Receive EOF")
                    if output_packets is not None:
                        output_packets.put(Packet.generate_eof_packet())
                    self.eof_ = True
                elif pkt.get_timestamp() != Timestamp.UNSET:
                    Log.log_node(LogLevel.DEBUG, task.get_node(), ' timestamp of this processing packet is: ',
                                 pkt.get_timestamp())
                    if output_packets is not None:
                        output_packets.put(pkt)

        if self.eof_:
            task.set_timestamp(Timestamp.DONE)
        return ProcessResult.OK
