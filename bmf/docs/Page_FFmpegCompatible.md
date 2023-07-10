## FFmpeg Fully Compatible

BMF is full compatible with capabilities of FFmpeg. FFmpeg is well known in the industry by rich features about AV demux, decode, filter, encode, mux, etc.
BMF implements buildin modules and combined flexible API to Python, C++, Go user to apply FFmpeg capabilities into their own solution.

### Environment
FFmpeg library is needed，the supported version and installation details can be found in Getting Started -> Install.


### Capabilities
- demux + decode
- filter
- encode + mux

The parameter details can be found in API->Built-in Decode Module, API->Built-in Filter Module and API->Built-in Encode Module.

### How To Use
Some typical examples will show the way to use BMF in media processing senario.
#### Decode Only
The example below decode a media file only:
```python
        import bmf

        input_video_path = "../files/test.mp4"

        graph = bmf.graph()

        stream = graph.decode({
            "input_path": input_video_path
        })

        (
            bmf.encode(
                stream['video'],
                stream['audio'],
                {
                    "null_output": 1
                }
            )
            .run()
        )
```

#### Transcode
In order to parallel modules, the encode module will be apply in scheduler 1 while decode and others in scheduler 0 by default.

```python
        import bmf

        input_video_path = "test.mp4"
        output_path = "./output.mp4"

        graph = bmf.graph({'dump_graph': 1})

        stream = graph.decode({
            "input_path": input_video_path,
            "dec_params": {
                "threads": "8"
            }
        })
        # using scale filter
        scaled = stream['video'].scale(320, 240)
        (
            bmf.encode(
                scaled,
                stream['audio'],
                {
                    "output_path": output_path,
                    "video_params": {
                        "codec": "h264",
                        "width": 320,
                        "height": 240,
                        "crf": 23,
                        "preset": "veryfast",
                    },
                    "audio_params": {
                        "codec": "aac",
                        "bit_rate": 128000,
                        "sample_rate": 44100,
                        "channels": 2
                    }
                }
            ).run()
        )
```

#### Image Encode
```python
        import bmf

        input_video_path = "test.png"
        output_path = "./image.jpg"

        (
            bmf.graph()
                .decode({'input_path': input_video_path})['video']
                .scale(320, 240)
                .encode(None, 
                    {
                        "output_path": output_path,
                        "format": "mjpeg",
                        "video_params": {
                            "codec": "jpg",
                            "width": 320,
                            "height": 240
                        }
                    }
                ).run()
        )
```

#### Stream Copy
```python
        import bmf

        input_path = "test.mp4"
        output_path = "./stream_copy.mp4"
        
        stream = bmf.graph().decode(
            {                                            'input_path': input_path,
                'video_codec': "copy"
            }
        )

        video_stream = stream['video']

        video_stream.encode(stream['audio'], {
            "output_path": "stream_copy.mp4",
        }).run()
```

#### Using FFmpeg Filters by Parameter
```python
    import bmf

    input_video_path1 = "test1.mp4"
    input_video_path2 = "test2.mp4"
    graph = bmf.graph({'dump_graph':1})
    stream1 = graph.decode({
        "input_path": input_video_path1
    })
    stream2 = graph.decode({
        "input_path": input_video_path2
    })
    #using "vstack" ffmpeg filter in a common way
    bmf.ff_filter([stream1, stream2], 'vstack', input=2).encode(None, {"output_path": "output.mp4"})
    graph.run()
```

#### Using Module Capability Directly (Sync Mode)
User can integrate thoes capabilities of module into their own project. For exp. to get a yuv frame decoded, or encode a yuv frame by calling encode()
```python
        import bmf

        input_video_path = "test.png"
        output_path = "output.jpg"

        # create decoder
        decoder = bmf_sync.sync_module("c_ffmpeg_decoder", {"input_path": input_video_path}, [], [0])

        '''
        # for non-builtin modules, use module_info instead of module_name to specify type/path/entry

        module_info = {
            "name": "my_module",
            "type": "",
            "path": "",
            "entry": ""
        }
        module = bmf_sync.sync_module(module_info, {"input_path": input_video_path}, [], [0])
        '''

        # create scale
        scale = bmf_sync.sync_module("c_ffmpeg_filter", {
            "name": "scale",                                                                                         "para": "320:240"
        }, [0], [0])
                                                                                                                 # create encoder
        encoder = bmf_sync.sync_module("c_ffmpeg_encoder", {
            "output_path": output_path,
            "format": "mjpeg",
            "video_params": {
                "codec": "jpg"
            }
        }, [0], [])
        # call init if necessary, otherwise we skip this step
        decoder.init()
        scale.init()
        encoder.init()

        # decode
        frames, _ = bmf_sync.process(decoder, None)
                                                                                                                 # scale                                                                                                  frames, _ = bmf_sync.process(scale, {0:frames[0]})

        # encode
        bmf_sync.process(encoder, {0:frames[0]})

        # send eof to encoder
        bmf_sync.send_eof(encoder)

        # call close if necessary, otherwise we skip this step
        decoder.close()                                                                                          scale.close()                                                                                            encoder.close()
```

#### Other Reference
There are also a lot of examples in test_transcode.py, test_sync_mode.py, etc. Please refer those sample code if needed.


### Tools
BMF provides some useful tools to help developer to debug, compare, quick verification, etc.
#### Run Graph
After the app run with "{'dump_graph': 1}", for example in [Transcode](#transcode), a json decription will be dump into a file original_graph.json as below:
```python
{
    "input_streams": [],
    "output_streams": [],
    "nodes": [
        {
            "module_info": {
                "name": "c_ffmpeg_decoder",
                "type": "",
                "path": "",
                "entry": ""
            },
            "meta_info": {
                "premodule_id": -1,
                "callback_binding": []
            },
            "option": {
                "input_path": "test.mp4",
                "dec_params": {
                    "threads": "8"
                }
            },
            "input_streams": [],
            "output_streams": [
                {
                    "identifier": "video:c_ffmpeg_decoder_0_1",
                    "stream_alias": ""
                },
                {
                    "identifier": "audio:c_ffmpeg_decoder_0_2",
                    "stream_alias": ""
                }
            ],
            "input_manager": "immediate",
            "scheduler": 0,
            "alias": "",
            "id": 0
        },
        {
            "module_info": {
                "name": "c_ffmpeg_filter",
                "type": "",
                "path": "",
                "entry": ""
            },
            "meta_info": {
                "premodule_id": -1,
                "callback_binding": []
            },
            "option": {
                "name": "scale",
                "para": "320:240"
            },
            "input_streams": [
                {
                    "identifier": "c_ffmpeg_decoder_0_1",
                    "stream_alias": ""
                }
            ],
            "output_streams": [
                {
                    "identifier": "c_ffmpeg_filter_1_0",
                    "stream_alias": ""
                }
            ],
            "input_manager": "immediate",
            "scheduler": 0,
            "alias": "",
            "id": 1
        },
        {
            "module_info": {
                "name": "c_ffmpeg_encoder",
                "type": "",
                "path": "",
                "entry": ""
            },
            "meta_info": {
                "premodule_id": -1,
                "callback_binding": []
            },
            "option": {
                "name": "scale",
                "para": "320:240"
            },
            "input_streams": [
                {
                    "identifier": "c_ffmpeg_decoder_0_1",
                    "stream_alias": ""
                }
            ],
            "output_streams": [
                {
                    "identifier": "c_ffmpeg_filter_1_0",
                    "stream_alias": ""
                }
            ],
            "input_manager": "immediate",
            "scheduler": 0,
            "alias": "",
            "id": 1
        },
        {
            "module_info": {
                "name": "c_ffmpeg_encoder",
                "type": "",
                "path": "",
                "entry": ""
            },
            "meta_info": {
                "premodule_id": -1,
                "callback_binding": []
            },
            "option": {
                "output_path": "./output.mp4",
                "video_params": {
                    "codec": "h264",
                    "width": 320,
                    "height": 240,
                    "crf": 23,
                    "preset": "veryfast"
                },
                "audio_params": {
                    "codec": "aac",
                    "bit_rate": 128000,
                    "sample_rate": 44100,
                    "channels": 2
                }
            },
            "input_streams": [
                {
                    "identifier": "c_ffmpeg_filter_1_0",
                    "stream_alias": ""
                },
                {
                    "identifier": "c_ffmpeg_decoder_0_2",
                    "stream_alias": ""
                }
            ],
            "output_streams": [],
            "input_manager": "immediate",
            "scheduler": 1,
            "alias": "",
            "id": 2
        }
    ],
    "option": {
        "dump_graph": 1
    },
    "mode": "Normal"
}
```

Users can check some details and modify it if they know what are they doing, and also the json graph can be run directly:
```bash
$ run_bmf_graph original_graph.json
```

#### Convert to FFmpeg Command Line
```python
import bmf

def run():
    my_graph = bmf.graph()
    my_graph.runFFmpegByConfig("original_graph.json")

run()
```
And a FFmpeg command line will be generated and run according to the json description:
```bash
ffmpeg -threads 8 -i test.mp4  -filter_complex "[0:v] scale=320:240[c_ffmpeg_filter_1_0]
" -map '[c_ffmpeg_filter_1_0]' -vcodec libx264 -pix_fmt yuv420p -crf 23 -preset veryfast -s 320x240 -map
0:a -acodec aac -b:a 128000 -ar 44100 -ac 2 -f mp4 ./output.mp4  -y
```