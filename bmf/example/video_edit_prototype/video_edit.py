from .common_process import segment_overlay, videos_concat, global_overlay
from .pre_process import preprocess_text_element, preprocess_image_element, preprocess_video_element, \
    preprocess_audio_element
from .util.preprocess_util import record_element_info, update_element_duration, update_element_size, \
    calculate_segment_duration, generate_background_image, generate_background_audio


def video_edit(segments, global_elements, output, graph):
    concat_segment_list = []

    for i, segment in enumerate(segments):
        element_list = []
        # overlay_video_stream = None

        if "BackGround" in segment and segment["BackGround"]:
            background_color = segment["BackGround"]
        else:
            background_color = '0xFFFFFFFF'

        # get initial element information
        # calculate and update startTime, duration, size according to params
        for element in segment['Elements']:
            if 'StartTime' not in element:
                element['StartTime'] = float(0)
            if not element['Type'] == 'text':
                record_element_info(element)
                update_element_duration(element)
            if not element['Type'] == 'audio':
                update_element_size(element, float(output['Width']), float(output['Height']))

        # calculate segment duration
        calculate_segment_duration(segment)

        # generate background with particular color
        background_image = generate_background_image(background_color, output['Width'], output['Height'], graph=graph)
        segment['background_stream'] = background_image

        # generate silent-audio for segment
        background_audio = generate_background_audio(segment['duration'], graph)
        segment['background_audio_stream'] = background_audio

        # element preprocess
        for j, element in enumerate(segment['Elements']):
            new_element = None
            if element['Type'] == 'text':
                new_element = preprocess_text_element(text_info=element, graph=graph)
            elif element['Type'] == 'image':
                new_element = preprocess_image_element(image_info=element, graph=graph)
            elif element['Type'] == 'video':
                new_element = preprocess_video_element(video_info=element, graph=graph)
            elif element['Type'] == 'audio':
                new_element = preprocess_audio_element(audio_info=element, graph=graph)
            if new_element:
                element_list.append(new_element)

        # save preprocessed-element-stream
        segment['Elements'] = element_list
        segment['width'], segment['height'] = float(output['Width']), float(output['Height'])

        # segment elements do overlay
        overlay_video_stream, overlay_audio_stream = segment_overlay(segment)

        concat_segment_info = dict()
        concat_segment_info['transition_mode'] = segment.get('Transition')
        concat_segment_info['transition_time'] = segment.get('TransitionTime')
        concat_segment_info['video_stream'] = overlay_video_stream
        concat_segment_info['audio_stream'] = overlay_audio_stream
        concat_segment_info['duration'] = segment['duration']
        concat_segment_info['elements'] = segment['Elements']

        concat_segment_list.append(concat_segment_info)

        # in order to dump_json, remove stream from element option
        for element in segment['Elements']:
            del element['stream']

    # concat all segments
    concat_stream = videos_concat(concat_list=concat_segment_list, output=output)

    # get initial global element information
    # calculate and update duration & size  according to params
    for element in global_elements:
        if 'StartTime' not in element:
            element['StartTime'] = float(0)
        if not element['Type'] == 'text':
            record_element_info(element)
            update_element_duration(element)
        if not element['Type'] == 'audio':
            update_element_size(element, float(output['Width']), float(output['Height']))

    # global element preprocess
    global_element_list = []
    for j, element in enumerate(global_elements):
        new_element = None
        if element['Type'] == 'text':
            new_element = preprocess_text_element(text_info=element, graph=graph)
        elif element['Type'] == 'image':
            new_element = preprocess_image_element(image_info=element, graph=graph)
        elif element['Type'] == 'video':
            new_element = preprocess_video_element(video_info=element, graph=graph)
        elif element['Type'] == 'audio':
            new_element = preprocess_audio_element(audio_info=element, graph=graph)
        if new_element:
            global_element_list.append(new_element)

    # get total duration
    total_duration = 0
    for segment in segments:
        total_duration += segment['duration']

    # do global overlay
    final_v_stream, final_a_stream = global_overlay(
        output['Width'], output['Height'], total_duration=total_duration, global_elements=global_element_list,
        source=concat_stream
    )

    # in order to dump_json, remove stream from element option
    for element in global_elements:
        del element['stream']

    return final_v_stream, final_a_stream
