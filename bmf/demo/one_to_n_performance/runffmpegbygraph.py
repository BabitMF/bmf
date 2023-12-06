import bmf
import time

def run():
    my_graph = bmf.graph()
    my_graph.runFFmpegByConfig("original_graph.json")

t1 = time.time()
run()
tlast = (time.time() - t1) * 1000
print("FFmpeg time cost (ms):", tlast)
