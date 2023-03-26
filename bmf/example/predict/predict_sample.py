import sys

sys.path.append("../../..")
import bmf
from bmf import LogLevel, Log


if __name__ == '__main__':
    Log.set_log_level(LogLevel.ERROR)

    # create sr module once
    sr_mod = bmf.create_module('onnx_sr', {
        "model_path": "v1.onnx"
    })

    # execute two tasks w/o loading model repeatedly
    for i in range(2):
        print('execute %dth task' % i)
        # build bmf graph and run
        (
            bmf.graph()
                .decode({'input_path': "../files/img_s.mp4"})['video']
                .module('onnx_sr', pre_module=sr_mod)
                .encode(None, {"output_path": "../files/out.mp4", "video_params":{"max_fr":30}})
                .run()
        )
