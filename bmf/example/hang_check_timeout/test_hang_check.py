import sys
import time
from bmf import LogLevel, Log
import bmf

sys.path.append("../../")

if __name__ == '__main__':
    Log.set_log_level(LogLevel.DEBUG)

    input_video_path = "../files/header.mp4"

    try:
        my_graph = bmf.graph()
        streams = my_graph.decode({'input_path': input_video_path})
        connect = bmf.module(streams, 'hang_module')
        my_graph.set_option({'time_out': 1.0})
        my_graph.run()
    except Exception as e:
        msg = str(e)
        print('Got bmf exception: ', msg)
