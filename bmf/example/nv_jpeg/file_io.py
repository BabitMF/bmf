      
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType

import bmf


class file_io(Module):
    def __init__(self, node, option=None):
        self.node_ = node
        self.save_dir_ = option["save_dir"]
        self.index = 0

    def process(self, task):
        input_queue = task.get_inputs()[0]

        while not input_queue.empty():
            pkt = input_queue.get()
            if pkt.get_timestamp() == Timestamp.EOF:
                task.set_timestamp(Timestamp.DONE)
            # TODO: add upload code here
            else:
                # print("upload get data")
                number_str = '{:05d}'.format(self.index)
                save_path = self.save_dir_+ "/" + number_str + ".jpg"
                print(save_path)
                avpacket = pkt.get(bmf.BMFAVPacket)
                data = avpacket.data.numpy()
                self.index = self.index+1
                with open(save_path, "wb") as fid:
                    fid.write(data)

        return ProcessResult.OK

    
