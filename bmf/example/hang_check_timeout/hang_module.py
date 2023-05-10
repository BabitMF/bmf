from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType
import time

class hang_module(Module):
    def __init__(self, node, option=None):
        self.node_ = node

    def reset(self):
        Log.log_node(LogLevel.DEBUG, self.node_, " is doing reset")

    def process(self, task):
        Log.log_node(LogLevel.DEBUG, self.node_, " process got input stream number: ", len(task.get_inputs().items()))
        Log.log_node(LogLevel.DEBUG, self.node_, " start to sleep")
        time.sleep(100)

        return ProcessResult.OK
