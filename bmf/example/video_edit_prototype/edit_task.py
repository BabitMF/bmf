from .video_edit import video_edit
import bmf
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
import time


def _edit_task(upload, output, segments=None, global_elements=None):
    # input param
    uploader = upload['Uploader']
    width = output['Width']
    height = output['Height']
    mode = output.get('Mode', 'normal')
    format = output.get('Format', 'mp4')
    segment_time = output.get('SegmentTime', 10)
    fps = output.get('Fps', 25)
    quality = output.get('Quality', 'medium')

    task_start = time.time()

    # create bmf graph
    graph = bmf.graph({"dump_graph": 1})

    # local output path
    output_path = "../files/video_edit_output.mp4"

    # result info
    result_info = dict()

    # incorrect param
    if width <= 0 or height <= 0:
        Log.log(LogLevel.ERROR, "output resolution error")
        return result_info
    if len(segments) <= 0:
        Log.log(LogLevel.ERROR, "merge_nodes is None")
        return result_info

    # process entrance
    final_video_stream, final_audio_stream = video_edit(segments=segments, global_elements=global_elements,
                                                        output=output, graph=graph)

    # encode and save in local
    final_video_stream.encode(final_audio_stream, {
        "output_path": output_path,
        "video_params": {
            "codec": "h264",
            "width": width,
            "height": height,
            "preset": "fast",
            "crf": "23",
            "x264-params": "ssim=1:psnr=1"
        },
        "audio_params": {
            "codec": "aac",
            "bit_rate": 128000,
            "sample_rate": 44100,
            "channels": 2
        },
        "mux_params": {
            "fflags": "+igndts",
            "movflags": "+faststart+use_metadata_tags",
            "max_interleave_delta": "0"
        }
    }).run()

    # TODO: upload result file

    # record task processing time
    task_end = time.time()
    task_time = round(task_end - task_start, 6)
    Log.log(LogLevel.INFO, "task_time: %f" % task_time)

    # result return
    result_info['output_path'] = output_path
    return result_info


def edit_task(request):
    body = request.body
    func_input = body['input']

    upload = func_input['Upload']
    output = func_input['Output']

    segments = func_input.get('Segments')
    global_elements = func_input.get('GlobalElements')

    return _edit_task(upload=upload, output=output, segments=segments, global_elements=global_elements)
