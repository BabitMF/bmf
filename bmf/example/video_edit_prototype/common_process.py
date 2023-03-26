import bmf

from .util.preprocess_util import generate_concat_option, generate_video_overlay_option, generate_audio_mix_option


def segment_overlay(segment):
    element_list = segment['Elements']

    # generate amix-stream and amix-option
    audio_stream_list, audio_option = generate_audio_mix_option(
        element_list, segment['duration'], source=segment['background_audio_stream']
    )

    # mix audio
    audio_stream = bmf.module(audio_stream_list, 'audio_mix', audio_option)

    # turn background image to video
    background_video_stream = segment['background_stream'].loop(loop=-1, start=0, size=1).trim(start=0,
                                                                                               duration=segment[
                                                                                                   'duration'])

    # generate overlay_stream and overlay_option
    overlay_stream, overlay_option = generate_video_overlay_option(
        segment['width'], segment['height'], segment['duration'], background_video_stream, element_list
    )

    # call 'video_overlay' to do overlay
    overlay_video_stream = (
        bmf.module(overlay_stream, 'video_overlay', overlay_option)
    )

    return overlay_video_stream, audio_stream


def videos_concat(concat_list=None, output=None):
    # generate concat_stream and concat_option
    concat_stream_list, concat_option = generate_concat_option(concat_list, output)

    #  call 'video_concat' to do concat
    concat_streams = (
        bmf.module(concat_stream_list, 'video_concat', concat_option)
    )

    return concat_streams


def global_overlay(out_width, out_height, total_duration=0, global_elements=None, source=None):
    # generate amix-stream and amix-option
    audio_stream_list, audio_option = generate_audio_mix_option(global_elements, total_duration, source=source[1])

    # mix audio
    global_audio_stream = bmf.module(audio_stream_list, 'audio_mix', audio_option)

    # generate overlay_stream and overlay_option
    overlay_stream, overlay_option = generate_video_overlay_option(out_width, out_height, total_duration, source[0],
                                                                   global_elements)

    # call 'video_overlay' to do overlay
    global_video_stream = (
        bmf.module(overlay_stream, 'video_overlay', overlay_option)
    )

    return global_video_stream, global_audio_stream
