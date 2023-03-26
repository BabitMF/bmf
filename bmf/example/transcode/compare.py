import sys
sys.path.append("../../..")
import bmf
from bmf import *


if __name__ == "__main__":
    import sys
    file_name = sys.argv[1]
    mode = sys.argv[2]
    graph = BmfGraph({})
    if mode == "ffmpeg":
        graph.runFFmpegByConfig(file_name)
    elif mode == "pythonEngine":
        graph.runPythonEngine(file_name)
    elif mode == "cEngine":
        graph.runCEngine(file_name)