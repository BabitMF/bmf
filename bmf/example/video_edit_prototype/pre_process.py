import bmf

text_image_num = 0


def preprocess_video_element(video_info, graph):
    # TODO: download from remote

    video = graph.decode({"input_path": video_info['Source']})

    # must have video stream
    video_v = video['video']

    # if no audio stream
    if 'audio_duration' not in video_info['initial_info']:
        processed_video = bmf.module([video_v], 'video_preprocess', video_info)
        video_info['stream'] = {
            'v': processed_video[0]
        }
    # both video and audio stream
    else:
        video_a = video['audio']
        processed_video = bmf.module([video_v, video_a], 'video_preprocess', video_info)
        video_info['stream'] = {
            'v': processed_video[0],
            'a': processed_video[1]
        }

    return video_info


def preprocess_image_element(image_info, graph):
    # TODO: download from remote

    image = graph.decode({"input_path": image_info['Source']})['video'].module('image_preprocess', image_info)

    image_info['stream'] = image

    return image_info


def preprocess_audio_element(audio_info, graph):
    # TODO: download from remote

    audio = graph.decode({"input_path": audio_info['Source']})['audio'].module('audio_preprocess', audio_info)

    audio_info['stream'] = audio

    return audio_info


def preprocess_text_element(text_info, graph):
    # image local path
    global text_image_num
    text_info['local_path'] = '../files/text_' + str(text_image_num) + '.png'
    text_image_num += 1

    text_image = graph.module('text_to_image', text_info).decode()['video']

    text_info['stream'] = text_image

    return text_info
