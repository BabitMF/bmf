from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType


class upload(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        pass

    def process(self, task):
        input_queue = task.get_inputs()[0]

        while not input_queue.empty():
            pkt = input_queue.get()
            if pkt.get_timestamp() == Timestamp.EOF:
                task.set_timestamp(Timestamp.DONE)
            # TODO: add upload code here
            else:
                # print("upload get data")
                # print(pkt.get_data())
                Log.log_node(LogLevel.DEBUG, self.node_, 'upload info', pkt.class_name)

        return ProcessResult.OK
