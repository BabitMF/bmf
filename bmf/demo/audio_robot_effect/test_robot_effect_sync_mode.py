import bmf
from bmf import bmf_sync, Packet
import os
import time

module_path = os.path.abspath(__file__)
py_module_info = {
    "name": "audio_robot_effect_module",
    "type": "",
    "path": module_path,
    "entry": "robot_effect:BMFRobotEffect"
}

def robot_effect(input_path, output_path):
    try:
        decoder = bmf_sync.sync_module(
        "c_ffmpeg_decoder", 
        {
            "input_path": input_path
        }, [], [0, 1])

        auido_effect = bmf_sync.sync_module(py_module_info,{}, [1], [1])

        encoder = bmf_sync.sync_module("c_ffmpeg_encoder", {
            "output_path": output_path,
            "format": "mp4",
            "audio_params": {
                "codec": "libfdk_aac",
                "bit_rate": 128000.0,
                "sample_rate": 44100,
                "channels": 2,
                "profile": "aac_low"
            }
        }, [0, 1], [])

        while True:
            frames, _ = bmf_sync.process(decoder, None)
            has_next = False
            for key in frames:
                if len(frames[key]) > 0:
                    has_next = True
                    break
            if not has_next:
                bmf_sync.send_eof(encoder)
                break
            if 0 in frames.keys() and len(frames[0]) > 0:
                bmf_sync.process(encoder, {0: frames[0]})
            if 1 in frames.keys() and len(frames[1]) > 0:
                processed_frames, _ = bmf_sync.process(auido_effect, {1: frames[1]})
                bmf_sync.process(encoder, {1: processed_frames[1]})
    except Exception as e:
        print(f"Error in process_audio: {e}")
        return False
    return True

def main():
    input_path = '../../files/counting_number.wav'
    output_path = './robot_effect.wav'
    start_time = time.time()
    robot_effect(input_path, output_path)
    print(f"\n============ process {input_path} time cost is {time.time() - start_time} ============")

if __name__ == "__main__":
    main()