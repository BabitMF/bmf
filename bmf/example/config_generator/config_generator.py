import bmf

input_video_path_1 = "../files/header.mp4"
input_video_path_2 = "../files/header.mp4"
input_video_path_3 = '../files/img.mp4'
logo_video_path_1 = "../files/xigua_prefix_logo_x.mov"
logo_video_path_2 = "../files/xigua_loop_logo2_x.mov"
output_path = "./video.mp4"

# some parameters
output_width = 1280
output_height = 720
logo_width = 320
logo_height = 144

# create graph
graph = bmf.graph()

# tail video
tail = graph.decode({'input_path': input_video_path_1})

# header video
header = graph.decode({'input_path': input_video_path_2})

# main video
video = graph.decode({'input_path': input_video_path_3})

# logo video
logo_1 = (
    graph.decode({'input_path': logo_video_path_1})['video']
        .scale(logo_width, logo_height)
)
logo_2 = (
    graph.decode({'input_path': logo_video_path_2})['video']
        .scale(logo_width, logo_height)
        .ff_filter('loop', loop=-1, size=991)
        .ff_filter('setpts', 'PTS+3.900/TB')
)

# main video processing
main_video = (
    video['video'].scale(output_width, output_height)
        .overlay(logo_1, repeatlast=0)
        .overlay(logo_2,
                    x='if(gte(t,3.900),960,NAN)',
                    y=0,
                    shortest=1)
)

# concat video
concat_video = (
    bmf.concat(header['video'].scale(output_width, output_height),
                main_video,
                tail['video'].scale(output_width, output_height),
                n=3)
)

# concat audio
concat_audio = (
    bmf.concat(header['audio'],
                video['audio'],
                tail['audio'],
                n=3, v=0, a=1)
)

bmf.encode(concat_video,
    concat_audio,
    {
        "output_path": output_path,
        "video_params": {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "preset": "veryfast",
            "crf": "23",
            "x264-params": "ssim=1:psnr=1"
        },
        "audio_params": {
            "codec": "aac",
            "bit_rate": 128000,
            "sample_rate": 48000,
            "channels": 2
        },
        "mux_params": {
            "fflags": "+igndts",
            "movflags": "+faststart+use_metadata_tags",
            "max_interleave_delta": "0"
        }
    })
    
graph.generate_config_file(file_name='generated_graph.json')
