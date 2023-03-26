from .ff_probe import ff_probe,get_video_duration,get_audio_duration,get_filetype,get_video_size

background_image_count = 0
background_audio_count = 0


def generate_background_image(background_color, width, height, graph):
    # TODO: directory is for test, use tempfile tool to save image in real situation
    global background_image_count
    background_image_file = '../files/background_' + str(background_image_count) + '.png'
    background_image_count += 1

    # construct background_image option
    background_image_option = {
        'width': width,
        'height': height,
        'background_color': background_color,
        'local_path': background_image_file
    }

    # call 'background_image' to create background image
    background_image = graph.module('background_image', background_image_option).decode()['video']
    return background_image


def generate_background_audio(segment_duration, graph):
    # TODO: directory is for test, use tempfile tool to save audio in real situation
    global background_audio_count
    background_audio_file = '../files/background_audio_' + str(background_audio_count) + '.aac'
    background_audio_count += 1

    # construct background_audio option
    background_audio_option = {
        'segment_duration': segment_duration,
        'local_path': background_audio_file
    }

    # call 'background_audio' to create background image
    background_audio = graph.module('background_audio', background_audio_option).decode()['audio']
    return background_audio


def update_image_option(image_info):
    source = image_info['Source']
    start_time = image_info.get('StartTime', 0)
    duration = image_info.get('Duration')
    position = image_info.get('Position', {
        'PosX': '100%',
        'PosY': '100%',
        'Width': '0',
        'Height': '0'
    })
    crop = image_info.get('Crop')
    rotate = image_info.get('Rotate', 0)
    vflip = image_info.get('Vflip', 0)
    hflip = image_info.get('Hflip', 0)
    border_radius = image_info.get('BorderRadius', '0')
    filters = image_info.get('Filters')
    extra_filters = image_info.get('ExtraFilters')

    option = {}
    option['vflip'] = 1
    return option


def update_video_option(video_info):
    source = video_info['Source']
    start_time = video_info.get('StartTime', 0)
    duration = video_info.get('Duration')
    position = video_info.get('Position', {
        'PosX': '100%',
        'PosY': '100%',
        'Width': '0',
        'Height': '0'
    })
    crop = video_info.get('Crop')
    speed = video_info.get('Speed', 1)
    rotate = video_info.get('Rotate', 0)
    delogo = video_info.get('Delogo')
    trims = video_info.get('Trims')
    vflip = video_info.get('Vflip', 0)
    hflip = video_info.get('Hflip', 0)
    border_radius = video_info.get('BorderRadius', '0')
    filters = video_info.get('Filters')
    extra_filters = video_info.get('ExtraFilters')
    volume = video_info.get('Volume', 0)
    mute = video_info.get('Mute', 0)

    option = {}
    option['vflip'] = 1
    return option


def record_element_info(element):
    # get info of element
    result_dict = ff_probe(element['Source'])
    initial_info = {}
    if element['Type'] == 'video':
        # duration of video_stream must be positive value
        video_duration = get_video_duration(result_dict)
        if video_duration == -1 or video_duration == 0:
            raise Exception('invalid video duration')
        else:
            initial_info['video_duration'] = video_duration

        # duration of audio_stream can be None
        audio_duration = get_audio_duration(result_dict)
        if audio_duration == -1:
            pass
        else:
            initial_info['audio_duration'] = audio_duration

        # size of video_stream must be positive value
        width, height = get_video_size(result_dict)
        if width == -1 or height == -1:
            raise Exception('invalid video width or height')
        else:
            initial_info['width'], initial_info['height'] = width, height

    elif element['Type'] == 'audio':
        # duration of audio_stream must be positive value
        audio_duration = get_audio_duration(result_dict)
        if audio_duration == -1:
            raise Exception('invalid audio duration')
        else:
            initial_info['audio_duration'] = audio_duration

    elif element['Type'] == 'image':
        # size of video_stream must be positive value
        width, height = get_video_size(result_dict)
        if width == -1 or height == -1:
            raise Exception('invalid image width or height')
        else:
            initial_info['width'], initial_info['height'] = width, height

    element['initial_info'] = initial_info


def update_element_duration(element):
    if element['Type'] == 'video':
        # update according to trims
        trims = element.get('Trims')
        if trims and len(trims) > 0:
            # if multi-cut interval invalid
            if float(trims[len(trims) - 1][1]) > element['initial_info']['video_duration']:
                raise Exception('multi-cut interval longer than initial video length')
            # record sum of trims interval
            trims_duration = 0
            for i in range(len(trims)):
                trim = trims[i]
                start = float(trim[0])
                end = float(trim[1])
                trims_duration += (end - start)
            # update video duration
            element['initial_info']['video_duration'] = trims_duration

        # update according to speed
        speed = element.get('Speed')
        if speed:
            # currently, supported speed interval is 0.5~2.0
            if float(speed) < 0.5 or float(speed) > 2.0:
                raise Exception('invalid speed, speed interval should range from 0.5 to 2.0')
            # update video_duration
            element['initial_info']['video_duration'] /= float(speed)

        # update according to duration
        given_v_duration = float(element.get('Duration'))
        if given_v_duration:
            # if given duration longer than real duration
            # (StartTime + Duration) can not longer than real duration
            if given_v_duration > element['initial_info']['video_duration']:
                given_v_duration = element['initial_info']['video_duration']
            # update video_duration
            element['initial_info']['video_duration'] = given_v_duration

        # if exists audio, audio duration must be as same as video
        if 'audio_duration' in element['initial_info']:
            element['initial_info']['audio_duration'] = element['initial_info']['video_duration']

    elif element['Type'] == 'audio':
        # update according to trims
        trims = element.get('Trims')
        if trims and len(trims) > 0:
            # if multi-cut interval invalid
            if float(trims[len(trims) - 1][1]) > element['initial_info']['audio_duration']:
                raise Exception('multi-cut interval longer than initial video length')
            # record sum of trims interval
            trims_duration = 0
            for i in range(len(trims)):
                trim = trims[i]
                start = trim[0]
                end = trim[1]
                trims_duration += (end - start)
            # update audio duration
            element['initial_info']['audio_duration'] = trims_duration

        # update according to speed
        speed = element.get('Speed')
        if speed:
            # currently, supported speed interval is 0.5~2.0
            if float(speed) < 0.5 or float(speed) > 2.0:
                raise Exception('invalid speed, speed interval should range from 0.5 to 2.0')
            # update audio_duration
            element['initial_info']['audio_duration'] /= float(speed)

        # update according to duration
        given_a_duration = element.get('Duration')
        if given_a_duration:
            # if given duration longer than real duration
            # (StartTime + Duration) can not longer than real duration
            if given_a_duration > element['initial_info']['audio_duration']:
                given_a_duration = element['initial_info']['audio_duration']
            # update video_duration
            element['initial_info']['audio_duration'] = given_a_duration


def update_element_size(element, out_width, out_height):
    # position param
    if 'Position' not in element:
        # if no given position, set default value
        element['Position'] = {
            'PosX': 0,
            'PosY': 0,
            'Width': out_width,
            'Height': out_height
        }
    else:
        position = element['Position']
        # turn percent-value to float-value
        if '%' in position['PosX']:
            position['PosX'] = float(position['PosX'][:-1])*out_width/float(100)
        if '%' in position['PosY']:
            position['PosY'] = float(position['PosY'][:-1])*out_height/float(100)
        if '%' in position['Width']:
            position['Width'] = float(position['Width'][:-1])*out_width/float(100)
        if '%' in position['Height']:
            position['Height'] = float(position['Height'][:-1])*out_height/float(100)
        # given size too large
        if float(position['PosX']) + float(position['Width']) > out_width or float(position['PosY']) + float(position['Height']) > out_height:
            raise Exception("invalid element position, input width or height too large")

    # text element has no crop param
    if element['Type'] == 'text':
        return
    # crop param of video or image element
    if 'Crop' in element:
        crop = element['Crop']
        # turn percent-value to float-value
        if '%' in crop['PosX']:
            crop['PosX'] = float(crop['PosX'][:-1])*float(element['Position']['Width'])/float(100)
        if '%' in crop['PosY']:
            crop['PosY'] = float(crop['PosY'][:-1])*float(element['Position']['Height'])/float(100)
        if '%' in crop['Width']:
            crop['Width'] = float(crop['Width'][:-1])*float(element['Position']['Width'])/float(100)
        if '%' in crop['Height']:
            crop['Height'] = float(crop['Height'][:-1])*float(element['Position']['Height'])/float(100)
        # given size too large
        if float(crop['PosX'] + crop['Width']) > element['Position']['Width'] or \
                float(crop['PosY'] + crop['Height']) > element['Position']['Height']:
            raise Exception("invalid element crop")


def calculate_segment_duration(segment):
    segment_duration = 0
    # find max ending moment as segment duration
    for element in segment['Elements']:
        # for video
        if element['Type'] == 'video':
            # ending moment = start time + element duration
            duration = element['StartTime'] + element['initial_info']['video_duration']
            if duration > segment_duration:
                segment_duration = duration
        # for audio
        elif element['Type'] == 'audio':
            # ending moment = start time + element duration
            duration = element['StartTime'] + element['initial_info']['audio_duration']
            if duration > segment_duration:
                segment_duration = duration
        # for image
        elif element['Type'] == 'image':
            # ending moment = start time + element duration
            duration = element['StartTime'] + element['Duration']
            if duration > segment_duration:
                segment_duration = duration
        # for text
        elif element['Type'] == 'text':
            # ending moment = start time + element duration
            duration = element['StartTime'] + element['Duration']
            if duration > segment_duration:
                segment_duration = duration

    # record segment duration
    segment['duration'] = segment_duration


def generate_audio_mix_option(element_list, total_duration=0, source=None):
    audio_stream_list = []
    audio_option = {}
    audios = []

    # for source audio in global-amix
    if source is not None:
        audio_stream_list.append(source)
        audios.append({
            "start": 0,
            "duration": total_duration
        })

    # for elements
    for element in element_list:
        if element['Type'] == 'video' and 'audio_duration' in element['initial_info']:
            audio_stream_list.append(element['stream']['a'])
            audios.append({
                "start": element['StartTime'],
                "duration": element['initial_info']['audio_duration']
            })
        if element['Type'] == 'audio':
            audio_stream_list.append(element['stream'])
            audios.append({
                "start": element['StartTime'],
                "duration": element['initial_info']['audio_duration']
            })

    audio_option['audios'] = audios
    return audio_stream_list, audio_option


def generate_video_overlay_option(out_width, out_height, out_duration, background, element_list):
    """
    Option example:
        option = {
            "source": {
                "start": 0,
                "duration": 5,
                "width": 640,
                "height": 480
            },
            "overlays": [
                {
                    "start": 0,
                    "duration": 2,
                    "width": 300,
                    "height": 200,
                    "pox_x": 0,
                    "pox_y": 0,
                    "loop": -1,
                    "repeat_last": 0
                },
                {
                    "start": 2,
                    "duration": 2,
                    "width": 300,
                    "height": 200,
                    "pox_x": 'W-300',
                    "pox_y": 0,
                    "loop": 0,
                    "repeat_last": 1
                }
            ]
        }
    """
    overlay_stream = []
    overlay_option = {}

    # set background as source stream
    overlay_stream.append(background)
    # add source to overlay_option
    source_option = {
        "start": 0,
        "duration": out_duration,
        "width": out_width,
        "height": out_height
    }
    overlay_option['source'] = source_option

    overlays = []
    for element in element_list:
        if element['Type'] == 'video':
            width = element['Position']['Width']
            height = element['Position']['Height']
            pos_x = element['Position']['PosX']
            pos_y = element['Position']['PosY']
            overlay = {
                "start": element.get('StartTime', 0),
                "duration": element['initial_info']['video_duration'],
                "width": width,
                "height": height,
                "pox_x": pos_x,
                "pox_y": pos_y,
                "loop": 0,
                "repeat_last": 0
            }
            overlays.append(overlay)
            overlay_stream.append(element['stream']['v'])

        elif element['Type'] == 'audio':
            pass

        elif element['Type'] == 'image':
            width = element['Position']['Width']
            height = element['Position']['Height']
            pos_x = element['Position']['PosX']
            pos_y = element['Position']['PosY']

            overlay = {
                "start": element.get('StartTime', 0),
                "duration": element['Duration'],
                "width": width,
                "height": height,
                "pox_x": pos_x,
                "pox_y": pos_y,
                "loop": 0,
                "repeat_last": 1
            }
            overlays.append(overlay)
            overlay_stream.append(element['stream'])

        elif element['Type'] == 'text':
            width = element['Position']['Width']
            height = element['Position']['Height']
            pos_x = element['Position']['PosX']
            pos_y = element['Position']['PosY']

            overlay = {
                "start": element.get('StartTime', 0),
                "duration": element['Duration'],
                "width": width,
                "height": height,
                "pox_x": pos_x,
                "pox_y": pos_y,
                "loop": 0,
                "repeat_last": 1
            }
            overlays.append(overlay)
            overlay_stream.append(element['stream'])

    overlay_option['overlays'] = overlays

    return overlay_stream, overlay_option


def generate_concat_option(concat_list, output):
    # get the unified size for concat segments
    width = output['Width']
    height = output['Height']

    concat_option = {"width": width, "height": height, "has_audio": 1}

    # segment option_list
    option_list = []
    # segment stream_list
    concat_stream_list = []

    for segment in concat_list:
        start = 0
        duration = segment['duration']
        transition_time = segment.get('TransitionTime', 0)
        transition_mode = segment.get('Transition', 1)
        option_list.append(
            {
                "start": start,
                "duration": duration,
                "transition_time": transition_time,
                "transition_mode": transition_mode
            }
        )

    concat_option["video_list"] = option_list

    # add stream to concat stream list
    for segment in concat_list:
        concat_stream_list.append(segment['video_stream'])
    for segment in concat_list:
        concat_stream_list.append(segment['audio_stream'])

    return concat_stream_list, concat_option
