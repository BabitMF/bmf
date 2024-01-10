import sys

sys.path.append("../../")
import bmf

sys.path.pop()

def test():
    input_video_path = "./controlnet/test_imgs/bird.png"
    input_prompt_path = "./prompt.txt"
    output_path = "./output.png"

    graph = bmf.graph()

    # dual inputs
    # -------------------------------------------------------------------------
    video = graph.decode({'input_path': input_video_path})
    prompt = video.module('text_module', {'path': input_prompt_path})

    control=bmf.module(streams=[video, prompt], module_info='controlnet_module')
    control.encode(None, {'output_path': output_path}).run()

    # sync mode
    # from bmf import bmf_sync, Packet
    # decoder = bmf_sync.sync_module("c_ffmpeg_decoder", {"input_path":"./ControlNet/test_imgs/bird.png"}, [], [0])
    # prompt = bmf_sync.sync_module('text_module', {'path': './prompt.txt'}, [], [1])
    # controlnet = bmf_sync.sync_module('controlnet_module', {}, [0, 1], [0])

    # decoder.init()
    # prompt.init()
    # controlnet.init()

    # img, _ = bmf_sync.process(decoder, None)
    # txt, _ = bmf_sync.process(prompt, None)
    # gen_img, _ = bmf_sync.process(controlnet, {0: img[0], 1: txt[1]})
    # --------------------------------------------------------------------------

    # video = graph.decode({
    #     "input_path": input_video_path,
    #     # "video_params": {
    #     #     "hwaccel": "cuda",
    #     #     # "pix_fmt": "yuv420p",
    #     # }
    # })
    # (video['video']
    #  .module('controlnet', {
        
    # })
    # .encode(
    #     None, {
    #         "output_path": output_path,
    #         "video_params": {
    #             "codec": "png",
    #         #     "pix_fmt": "cuda",
    #         }
    #     }).run())


if __name__ == '__main__':
    test()
