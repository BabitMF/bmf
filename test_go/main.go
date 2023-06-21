package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"time"

	"code.byted.org/videoarch/bmf-gosdk/bmf"
	"code.byted.org/videoarch/bmf-gosdk/builder"
)

func normalMode() {
	fmt.Println("Normal mode")

	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": "./output_normal_mode.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testAudio() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(nil, vid.Stream(1), map[string]interface{}{
		"output_path": "./audio.mp4",
		"audio_params": map[string]interface{}{
			"codec":       "aac",
			"bit_rate":    128000,
			"sample_rate": 44100,
			"channels":    2,
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testWithInputOnlyAudio() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/only_audio.mp4",
	}, nil)
	g.Encode(nil, vid.Stream(1), map[string]interface{}{
		"output_path": "./output.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testWithEncodeWithAudioStreamButNoAudioFrame() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/only_video.mp4",
	}, nil)
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": "./output.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testWithNullAudio() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	audioStream := g.FFmpegFilter(nil, "anullsrc", map[string]interface{}{
		"r":  "48000",
		"cl": "2",
	}).Atrim(map[string]interface{}{
		"start": "0",
		"end":   "6",
	}).Stream(0)
	g.Encode(vid.Stream(0), audioStream, map[string]interface{}{
		"output_path": "./with_null_audio.mp4",
		"video_params": map[string]interface{}{
			"codec":  "h264",
			"width":  320,
			"height": 240,
			"crf":    23,
			"preset": "veryfast",
		},
		"audio_params": map[string]interface{}{
			"codec":       "aac",
			"bit_rate":    128000,
			"sample_rate": 44100,
			"channels":    2,
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testSimple() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": "./simple.mp4",
		"video_params": map[string]interface{}{
			"psnr":   1,
			"codec":  "h264",
			"width":  320,
			"height": 240,
			"crf":    23,
			"preset": "veryfast",
		},
		"audio_params": map[string]interface{}{
			"codec":       "aac",
			"bit_rate":    128000,
			"sample_rate": 44100,
			"channels":    2,
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testhls() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": "./out.hls",
		"format":      "hls",
		"mux_params": map[string]interface{}{
			"hls_list_size":        "0",
			"hls_time":             "10",
			"hls_segment_filename": "./file%03d.ts",
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testCrypt() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path":     "../../../bmf/example/files/encrypt.mp4",
		"decryption_key": "b23e92e4d603468c9ec7be7570b16229",
	}, nil)
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": "./crypt.mp4",
		"mux_params": map[string]interface{}{
			"encryption_scheme": "cenc-aes-ctr",
			"encryption_key":    "76a6c65c5ea762046bd749a2e632ccbb",
			"encryption_kid":    "a7e61c373e219033c21091fa607bf3b8",
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testOption() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
		"start_time": 2,
	}, nil)
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": "./option.mp4",
		"video_params": map[string]interface{}{
			"codec":       "h264",
			"width":       1280,
			"height":      720,
			"crf":         23,
			"preset":      "veryfast",
			"x264-params": "ssim=1:psnr=1",
		},
		"audio_params": map[string]interface{}{
			"codec":       "aac",
			"bit_rate":    128000,
			"sample_rate": 44100,
			"channels":    2,
		},
		"mux_params": map[string]interface{}{
			"fflags":               "+igndts",
			"movflags":             "+faststart+use_metadata_tags",
			"max_interleave_delta": "0",
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testImage() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/overlay.png",
	}, nil)
	vid.Stream(0).Scale("320:240").Encode(nil, map[string]interface{}{
		"output_path": "./image.jpg",
		"format":      "mjpeg",
		"video_params": map[string]interface{}{
			"codec":  "jpg",
			"width":  320,
			"height": 240,
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testVideo() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	tail := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/header.mp4",
	}, nil)

	header := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/header.mp4",
	}, nil)

	video := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)

	logo_1 := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/xigua_prefix_logo_x.mov",
	}, nil).Stream(0).Scale("320:144")

	logo_2 := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/xigua_loop_logo2_x.mov",
	}, nil).Stream(0).Scale("320:144").Loop(map[string]interface{}{
		"loop": -1,
		"size": 991,
	}).Setpts("PTS+3.900/TB")

	main_video := video.Stream(0).Scale("1280:720").Overlay([]*builder.BMFStream{logo_1.Stream(0)}, map[string]interface{}{
		"repeatlast": 0,
	}).Overlay([]*builder.BMFStream{logo_2.Stream(0)}, map[string]interface{}{
		"x":        "if(gte(t,3.900),960,NAN)",
		"y":        0,
		"shortest": 1,
	})

	concat_video := g.Concat([]*builder.BMFStream{header.Stream(0).Scale("1280:720").Stream(0), main_video.Stream(0), tail.Stream(0).Scale("1280:720").Stream(0)}, map[string]interface{}{
		"n": 3,
	})
	concat_audio := g.Concat([]*builder.BMFStream{header.Stream(1), video.Stream(1), tail.Stream(1)}, map[string]interface{}{
		"n": 3,
		"v": 0,
		"a": 1,
	})
	g.Encode(concat_video.Stream(0), concat_audio.Stream(0), map[string]interface{}{
		"output_path": "./video.mp4",
		"video_params": map[string]interface{}{
			"codec":       "h264",
			"width":       1280,
			"height":      720,
			"crf":         23,
			"preset":      "veryfast",
			"x264-params": "ssim=1:psnr=1",
			"vsync":       "vfr",
			"max_fr":      60,
		},
		"audio_params": map[string]interface{}{
			"codec":       "aac",
			"bit_rate":    128000,
			"sample_rate": 48000,
			"channels":    2,
		},
		"mux_params": map[string]interface{}{
			"fflags":               "+igndts",
			"movflags":             "+faststart+use_metadata_tags",
			"max_interleave_delta": "0",
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testConcatVideoAndAudio() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	video1 := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)

	video2 := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)

	concat_video := g.Concat([]*builder.BMFStream{video1.Stream(0), video2.Stream(0)}, nil)
	concat_audio := g.Concat([]*builder.BMFStream{video1.Stream(1), video2.Stream(1)}, map[string]interface{}{
		"v": 0,
		"a": 1,
	})
	g.Encode(concat_video.Stream(0), concat_audio.Stream(0), map[string]interface{}{
		"output_path": "./concat_video_and_audio.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testShortVideoConcat() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	video := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil).Stream(0)

	video2 := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/single_frame.mp4",
	}, nil).Stream(0)

	vout := video.Concat([]*builder.BMFStream{video2}, nil)
	g.Encode(vout.Stream(0), nil, map[string]interface{}{
		"output_path": "./simple.mp4",
		"video_params": map[string]interface{}{
			"codec":  "h264",
			"width":  320,
			"height": 240,
			"crf":    23,
			"preset": "veryfast",
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testMapParam() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	input_video_path := "../../../bmf/example/files/big_bunny_multi_stream.mp4"
	output_path_1 := "./output_1.mp4"
	output_path_2 := "./output_2.mp4"
	video1 := g.Decode(map[string]interface{}{
		"input_path": input_video_path,
		"map_v":      0,
		"map_a":      0,
	}, nil)

	g.Encode(video1.Stream(0), video1.Stream(1), map[string]interface{}{
		"output_path": output_path_1,
	})

	video2 := g.Decode(map[string]interface{}{
		"input_path": input_video_path,
		"map_v":      1,
		"map_a":      1,
	}, nil)

	g.Encode(video2.Stream(0), video2.Stream(1), map[string]interface{}{
		"output_path": output_path_2,
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testRGB2Video() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/test_rgba_806x654.rgb",
		"s":          "806:654",
		"pix_fmt":    "rgba",
	}, nil)
	videoStream := vid.Stream(0).Loop(map[string]interface{}{
		"loop": 50,
		"size": 1,
	})
	g.Encode(videoStream.Stream(0), nil, map[string]interface{}{
		"output_path": "./rgb2video.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testStreamCopy() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path":  "../../../bmf/example/files/img.mp4",
		"video_codec": "copy",
	}, nil)
	videoStream := vid.Stream(0)
	videoStream.Encode(vid.Stream(1), map[string]interface{}{
		"output_path": "./stream_copy.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testStreamAudioCopy() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path":  "../../../bmf/example/files/live_audio.flv",
		"video_codec": "copy",
		"audio_codec": "copy",
	}, nil)
	g.Encode(nil, vid.Stream(1), map[string]interface{}{
		"output_path": "./audio_copy.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testExtractFrames() {
	g := builder.NewBMFGraph(builder.Generator, nil)
	vstream := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
		"video_params": map[string]interface{}{
			"extract_frames": map[string]interface{}{
				"fps": 0.5,
			},
		},
	}, nil).Stream(0)

	nextPkt := vstream.Start()

	num := 0
	for true {
		pkt := nextPkt()
		ts := pkt.Timestamp()
		if ts == bmf.EOF {
			g.Close()
			break
		}
		videoFrame, _ := pkt.GetVideoFrame()
		if videoFrame != nil {
			num++
		}
	}
	if num != 5 {
		panic(nil)
	}
	fmt.Printf("num = %d", num)
}

func testIncorrectStreamNotify() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	defer func() {
		if panicInfo := recover(); panicInfo != nil {
			fmt.Printf("%v", panicInfo)
			return
		}
	}()
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	streamNotify := 0.1
	v := vid.Stream(streamNotify)
	g.Encode(v, nil, map[string]interface{}{
		"output_path": "./incorrect_stream_notify.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testIncorrectEncoderParam() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	defer func() {
		if panicInfo := recover(); panicInfo != nil {
			fmt.Printf("%v", panicInfo)
			return
		}
	}()
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	v := vid.Stream(0)
	a := vid.Stream(1)
	wrong_k_1 := "wrong_key_1"
	wrong_v_1 := "wrong_value_1"
	wrong_k_2 := "wrong_key_2"
	wrong_v_2 := "wrong_value_2"
	g.Encode(v, a, map[string]interface{}{
		"output_path": "./incorrect_encoder_param.mp4",
		"video_params": map[string]interface{}{
			"codec":   "h264",
			"crf":     23,
			wrong_k_1: wrong_v_1,
			wrong_k_2: wrong_v_2,
		},
		"audio_params": map[string]interface{}{
			wrong_k_1: wrong_v_1,
			wrong_k_2: wrong_v_2,
		},
		"mux_params": map[string]interface{}{
			wrong_k_1: wrong_v_1,
			wrong_k_2: wrong_v_2,
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testDurations() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
		"durations":  []float64{1.5, 3, 5, 6},
	}, nil)
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": "./durations.mp4",
		"video_params": map[string]interface{}{
			"codec":  "h264",
			"width":  320,
			"height": 240,
			"crf":    23,
			"preset": "veryfast",
			"vsync":  "vfr",
			"r":      30,
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testOutputRawVideo() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(vid.Stream(0), nil, map[string]interface{}{
		"output_path": "./out.yuv",
		"video_params": map[string]interface{}{
			"codec": "rawvideo",
		},
		"format": "rawvideo",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testOutputNull() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"null_output": 1,
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testVframes() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(vid.Stream(0), nil, map[string]interface{}{
		"output_path": "./simple.mp4",
		"vframes":     30,
		"video_params": map[string]interface{}{
			"codec":  "h264",
			"width":  640,
			"height": 480,
			"crf":    23,
			"preset": "veryfast",
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testEncoderPushOutputMp4() {
	g := builder.NewBMFGraph(builder.Generator, nil)
	vstream := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil).Stream(0)

	fName := "./simple_vframe_python.mp4"

	encoderStream := g.Encode(vstream, nil, map[string]interface{}{
		"output_path": fName,
		"push_output": 1,
		"vframes":     60,
		"video_params": map[string]interface{}{
			"codec":  "h264",
			"width":  640,
			"height": 480,
			"crf":    23,
			"preset": "veryfast",
		},
	}).Stream(0)

	nextPkt := encoderStream.Start()

	file, err := os.OpenFile(
		fName,
		os.O_WRONLY|os.O_TRUNC|os.O_CREATE,
		0666,
	)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	for true {
		pkt := nextPkt()
		ts := pkt.Timestamp()
		if ts == bmf.EOF {
			fmt.Println("generator mode meet EOF, ready to close")
			g.Close()
			break
		}
		avPkt, err := pkt.GetBMFAVPacket()
		if err != nil {
			fmt.Println("GetBMFAVPacket error\n")
		}
		offset, _ := avPkt.GetOffset()
		whence, _ := avPkt.GetWhence()
		data, _ := avPkt.DataPtr()
		if offset > 0 {
			file.Seek(offset, int(whence))
		}
		file.Write(data)
		if err != nil {
			panic(err)
		}
	}
}

func testEncoderPushOutputImage2Pipe() {
	g := builder.NewBMFGraph(builder.Generator, nil)
	vstream := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil).Stream(0)

	vframes_num := 2

	encoderStream := g.Encode(vstream, nil, map[string]interface{}{
		"push_output":      1,
		"vframes":          vframes_num,
		"format":           "image2pipe",
		"avio_buffer_size": 65536,
		"video_params": map[string]interface{}{
			"codec":  "jpg",
			"width":  640,
			"height": 480,
			"crf":    23,
			"preset": "veryfast",
		},
	}).Stream(0)

	nextPkt := encoderStream.Start()

	write_num := 0
	for true {
		pkt := nextPkt()
		if write_num < vframes_num {
			ts := pkt.Timestamp()
			if ts == bmf.EOF {
				break
			}

			avPkt, err := pkt.GetBMFAVPacket()
			if err != nil {
				fmt.Println("GetBMFAVPacket error\n")
			}

			data, _ := avPkt.DataPtr()
			sprintf := fmt.Sprintf("simple_image%d.jpg", write_num)
			err = ioutil.WriteFile(sprintf, data, 0644)
			if err != nil {
				fmt.Println("write file error")
			}
			write_num += 1
		} else {
			g.Close()
			break
		}
	}
}

func testEncoderPushOutputAudioPcmS16le() {
	g := builder.NewBMFGraph(builder.Generator, nil)
	aStream := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil).Stream(1)

	fName := "./test_audio_simple_pcm_s16le.wav"

	encoderStream := g.Encode(nil, aStream, map[string]interface{}{
		"output_path": fName,
		"format":      "wav",
		"push_output": 1,
		"audio_params": map[string]interface{}{
			"codec": "pcm_s16le",
		},
	}).Stream(0)

	nextPkt := encoderStream.Start()

	file, err := os.OpenFile(
		fName,
		os.O_WRONLY|os.O_TRUNC|os.O_CREATE,
		0666,
	)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	for true {
		pkt := nextPkt()
		ts := pkt.Timestamp()
		if ts == bmf.EOF {
			fmt.Println("generator mode meet EOF, ready to close")
			g.Close()
			break
		}
		avPkt, err := pkt.GetBMFAVPacket()
		if err != nil {
			fmt.Println("GetBMFAVPacket error\n")
		}
		offset, _ := avPkt.GetOffset()
		whence, _ := avPkt.GetWhence()
		data, _ := avPkt.DataPtr()
		if offset > 0 {
			file.Seek(offset, int(whence))
		}
		file.Write(data)
		if err != nil {
			panic(err)
		}
	}
}

func testSkipFrame() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
		"skip_frame": 32,
	}, nil)
	g.Encode(vid.Stream(0), nil, map[string]interface{}{
		"output_path": "./test_skip_frame_video.mp4",
		"video_params": map[string]interface{}{
			"codec":  "h264",
			"crf":    23,
			"preset": "veryfast",
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testSegmentTrans() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path":  "../../../bmf/example/files/img.mp4",
		"video_codec": "copy",
	}, nil)
	output_path := "./simple_%05d.mp4"
	g.Encode(vid.Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": output_path,
		"format":      "segment",
		"mux_params": map[string]interface{}{
			"segment_time": 4,
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testAudioModule() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(nil, vid.Stream(1).Module(nil, "my_module", builder.Python, "", "", map[string]interface{}{}, nil).Stream(0), map[string]interface{}{
		"output_path": "./audio_c_module.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testExceptionInPythonModule() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	defer func() {
		if panicInfo := recover(); panicInfo != nil {
			fmt.Printf("%v", panicInfo)
			return
		}
	}()
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(nil, vid.Stream(1).Module(nil, "my_module", builder.Python, "", "", map[string]interface{}{"exception": 1}, nil).Stream(0), map[string]interface{}{
		"output_path": "./test_exception_in_python_module.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testVideoOverlays() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	logo_path := "../../../bmf/example/files/xigua_prefix_logo_x.mov"
	output_path := "./overlays.mp4"

	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	logo_1 := g.Decode(map[string]interface{}{
		"input_path": logo_path,
	}, nil).Stream(0)
	output_streams := vid.Stream(0).Scale("1280:720").Trim("start=0:duration=7").Setpts("PTS-STARTPTS")
	overlay := logo_1.Scale("300:200").Loop("loop=0:size=10000").Setpts("PTS+0/TB")

	output_streams.Overlay([]*builder.BMFStream{overlay.Stream(0)}, "x=if(between(t,0,7),0,NAN):y=if(between(t,0,7),0,NAN):repeatlast=1").Encode(vid.Stream(1), map[string]interface{}{
		"output_path": output_path,
		"video_params": map[string]interface{}{
			"width":  640,
			"height": 480,
			"codec":  "h264",
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func testVideoConcat() {
	g := builder.NewBMFGraph(builder.Normal, nil)
	output_path := "./video_concat.mp4"
	var video_concat_streams []*builder.BMFStream
	var video_transit_streams []*builder.BMFStream
	var audio_concat_streams []*builder.BMFStream
	for i := 0; i < 3; i++ {
		video := g.Decode(map[string]interface{}{
			"input_path": "../../../bmf/example/files/img.mp4",
		}, nil)
		video_stream := video.Stream(0).Scale("1280:720")

		if i < 2 {
			split_stream := video_stream.Split("")
			concat_stream := split_stream.Stream(0).Trim("start=0:duration=5").Setpts("PTS-STARTPTS")
			transition_stream := split_stream.Stream(1).Trim("start=5:duration=2").Setpts("PTS-STARTPTS").Scale("200:200")

			video_transit_streams = append(video_transit_streams, transition_stream.Stream(0))
			video_concat_streams = append(video_concat_streams, concat_stream.Stream(0))
		} else {
			concat_stream := video_stream.Trim("start=0:duration=5").Setpts("PTS-STARTPTS")
			video_concat_streams = append(video_concat_streams, concat_stream.Stream(0))
		}

		if i > 0 {
			concat_stream := video_concat_streams[i].Overlay([]*builder.BMFStream{video_transit_streams[i-1]}, "repeatlast=0")
			video_concat_streams = video_concat_streams[:len(video_concat_streams)-1]
			video_concat_streams = append(video_concat_streams, concat_stream.Stream(0))
		}
		audio_stream := video.Stream(1).Atrim("start=0:duration=5").Asetpts("PTS-STARTPTS").Afade("t=in:st=0:d=2").Afade("t=out:st=5:d=2")

		audio_concat_streams = append(audio_concat_streams, audio_stream.Stream(0))
	}
	concat_video := g.Concat(video_concat_streams, "n=3:v=1:a=0").Stream(0)
	concat_audio := g.Concat(audio_concat_streams, "n=3:v=0:a=1").Stream(0)
	g.Encode(concat_video, concat_audio, map[string]interface{}{
		"output_path": output_path,
		"video_params": map[string]interface{}{
			"width":  1280,
			"height": 720,
		},
	})
	gInst := g.Run(true)
	gInst.Close()
}

func premoduleMode() {
	fmt.Println("PreModule mode")
	m, _ := builder.NewCBMFModule("pass_through", nil, builder.Cpp, "", "")
	g := builder.NewBMFGraph(builder.Normal, nil)
	vid := g.Decode(map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, nil)
	g.Encode(vid.Stream(0).Module(nil, "pass_through", builder.Cpp, "", "", nil, m).Stream(0), vid.Stream(1), map[string]interface{}{
		"output_path": "./output_pre_module.mp4",
	})
	gInst := g.Run(true)
	gInst.Close()
}

func syncMode() {
	done := false
	decoder, err0 := bmf.NewModuleFunctorBuiltin("c_ffmpeg_decoder", map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, 0, 1)
	if decoder == nil {
		fmt.Printf("Load decoder module failed %v\n", err0)
	} else {
		fmt.Printf("Load decoder module successful\n")
	}

	encoder, err1 := bmf.NewModuleFunctorBuiltin("c_ffmpeg_encoder", map[string]interface{}{
		"output_path": "./output_sync_mode.mp4",
	}, 1, 0)
	if encoder == nil {
		fmt.Printf("Load encoder module failed %v\n", err1)
	} else {
		fmt.Printf("Load encoder module successful\n")
	}

	pass, err2 := bmf.NewModuleFunctorBuiltin("pass_through", nil, 1, 1)
	if pass == nil {
		fmt.Printf("Load pass_through module failed %v\n", err2)
	} else {
		fmt.Printf("Load pass_through module successful\n")
	}
	passChan := make(chan []*bmf.Packet, 10)
	encChan := make(chan []*bmf.Packet, 10)
	fmt.Println("Sync mode")
	// PassThrough
	go func() {
		for opkts_dec := range passChan {
			eofSet := false

			// do pass_through module
			for i := 0; i < len(opkts_dec); i++ {
				ipkts_pass := []*bmf.Packet{}
				ipkts_pass = append(ipkts_pass, opkts_dec[i])
				if opkts_dec[i].Timestamp() == bmf.EOF {
					eofSet = true
				}
				_, err2 := pass.Execute(ipkts_pass, true)
				if err2 != nil {
					fmt.Printf("Pass Through failed! %v\n", err2)
					break
				}

				opkts_pass, err1 := pass.Fetch(0)
				if err1 != nil {
					fmt.Printf("Pass Through Fetch failed! error : %v\n", err1)
					break
				}
				// do encoder module
				for i := 0; i < len(opkts_pass); i++ {
					encChan <- []*bmf.Packet{opkts_pass[i]}
				}

			}

			if eofSet {
				fmt.Println("pass done")
				encChan <- []*bmf.Packet{bmf.GenerateEofPacket()}
				pass.Free()
				break
			}
		}
	}()

	// Encoder
	go func() {
		for opkts_pass := range encChan {
			eofSet := false

			// do encoder module
			for i := 0; i < len(opkts_pass); i++ {
				_, err := encoder.Call([]*bmf.Packet{opkts_pass[i]})
				if err != nil {
					fmt.Printf("encoder call failed! error : %v\n", err)
					break
				}
				if opkts_pass[i].Timestamp() == bmf.EOF {
					eofSet = true
				}
			}

			if eofSet {
				fmt.Println("enc done")
				encoder.Free()
				done = true
				break
			}
		}
	}()

	for {
		// do decoder module
		is_done, err := decoder.Execute([]*bmf.Packet{}, true)
		if is_done {
			fmt.Printf("Decode done\n")

			// EOF packet
			p_eof := bmf.GenerateEofPacket()
			pkts_eof := []*bmf.Packet{}
			pkts_eof = append(pkts_eof, p_eof)
			passChan <- pkts_eof
			decoder.Free()
			break
		}

		if err != nil {
			fmt.Printf("decoder execute failed! error : %v\n", err)
			break
		}

		opkts_dec, err1 := decoder.Fetch(0)
		if err1 != nil {
			fmt.Printf("decoder fetch failed!\n, error : %v", err1)
			break
		}

		// do pass_through module
		for i := 0; i < len(opkts_dec); i++ {
			ipkts_pass := []*bmf.Packet{}
			ipkts_pass = append(ipkts_pass, opkts_dec[i])
			passChan <- ipkts_pass
		}
	}

	for !done {
		time.Sleep(100 * time.Millisecond)
	}

}

func syncModeSerial() {
	decoder, err0 := bmf.NewModuleFunctorBuiltin("c_ffmpeg_decoder", map[string]interface{}{
		"input_path": "../../../bmf/example/files/img.mp4",
	}, 0, 1)
	defer decoder.Free()
	if decoder == nil {
		fmt.Printf("Load decoder module failed %v\n", err0)
	} else {
		fmt.Printf("Load decoder module successful\n")
	}

	encoder, err1 := bmf.NewModuleFunctorBuiltin("c_ffmpeg_encoder", map[string]interface{}{
		"output_path": "./output_sync_mode_serial.mp4",
	}, 1, 0)
	defer encoder.Free()
	if encoder == nil {
		fmt.Printf("Load encoder module failed %v\n", err1)
	} else {
		fmt.Printf("Load encoder module successful\n")
	}

	pass, err2 := bmf.NewModuleFunctorBuiltin("pass_through", nil, 1, 1)
	defer pass.Free()
	if pass == nil {
		fmt.Printf("Load pass_through module failed %v\n", err2)
	} else {
		fmt.Printf("Load pass_through module successful\n")
	}
	fmt.Println("Sync mode Serial")

	for {
		// do decoder module
		is_done, err := decoder.Execute([]*bmf.Packet{}, true)
		if is_done {
			fmt.Printf("Decode done\n")

			// EOF packet
			p_eof := bmf.GenerateEofPacket()
			pkts_eof := []*bmf.Packet{}
			pkts_eof = append(pkts_eof, p_eof)
			is_done_eof, err_eof := pass.Execute(pkts_eof, true)
			if is_done_eof {
				break
			}
			if err_eof != nil {
				fmt.Println("PassThrough EOF execute failed error : %v\n", err_eof)
			}
			opkts_eof, err1_eof := pass.Fetch(0)
			if err1_eof != nil {
				fmt.Println("PassThrough EOF fetch failed error : %v\n", err1_eof)
			}

			for i := 0; i < len(opkts_eof); i++ {
				_, err_enc_eof := encoder.Call([]*bmf.Packet{opkts_eof[i]})
				if err_enc_eof != nil {
					fmt.Printf("encoder EOF call failed! error : %v\n", err_enc_eof)
				}
			}
			break
		}

		if err != nil {
			fmt.Printf("decoder execute failed! error : %v\n", err)
			break
		}

		opkts_dec, err1 := decoder.Fetch(0)
		if err1 != nil {
			fmt.Printf("decoder fetch failed!\n, error : %v", err1)
			break
		}

		// do pass_through module
		for i := 0; i < len(opkts_dec); i++ {
			ipkts_pass := []*bmf.Packet{}
			ipkts_pass = append(ipkts_pass, opkts_dec[i])
			_, err2 := pass.Execute(ipkts_pass, true)
			if err2 != nil {
				fmt.Printf("Pass Through failed! %v\n", err2)
				break
			}

			opkts_pass, err1 := pass.Fetch(0)
			if err1 != nil {
				fmt.Printf("Pass Through Fetch failed! error : %v\n", err1)
				break
			}

			// do encoder module
			for i := 0; i < len(opkts_pass); i++ {
				_, err := encoder.Call([]*bmf.Packet{opkts_pass[i]})
				if err != nil {
					fmt.Printf("encoder call failed! error : %v\n", err)
				}
			}

		}
	}
}

func testSyncVideoFrame() {
	decoder, err0 := bmf.NewModuleFunctorBuiltin("c_ffmpeg_decoder", map[string]interface{}{
		"input_path": "../../../bmf/example/files/overlay.png",
	}, 0, 1)
	defer decoder.Free()
	if decoder == nil {
		fmt.Printf("Load decoder module failed %v\n", err0)
	} else {
		fmt.Printf("Load decoder module successful\n")
	}

	encoder, err1 := bmf.NewModuleFunctorBuiltin("c_ffmpeg_encoder", map[string]interface{}{
		"output_path": "./videoframe.jpg",
		"format":      "mjpeg",
		"video_params": map[string]interface{}{
			"codec": "jpg",
		},
	}, 1, 0)
	defer encoder.Free()
	if encoder == nil {
		fmt.Printf("Load encoder module failed %v\n", err1)
	} else {
		fmt.Printf("Load encoder module successful\n")
	}

	pass, err2 := bmf.NewModuleFunctorBuiltin("c_ffmpeg_filter", map[string]interface{}{
		"name": "scale",
		"para": "320:240",
	}, 1, 1)
	defer pass.Free()
	if pass == nil {
		fmt.Printf("Load pass_through module failed %v\n", err2)
	} else {
		fmt.Printf("Load pass_through module successful\n")
	}

	for {
		// do decoder module
		is_done, err := decoder.Execute([]*bmf.Packet{}, true)
		if is_done {
			fmt.Printf("Decode done\n")

			// EOF packet
			p_eof := bmf.GenerateEofPacket()
			pkts_eof := []*bmf.Packet{}
			pkts_eof = append(pkts_eof, p_eof)
			is_done_eof, err_eof := pass.Execute(pkts_eof, true)
			if is_done_eof {
				break
			}
			if err_eof != nil {
				fmt.Println("PassThrough EOF execute failed error : %v\n", err_eof)
			}
			opkts_eof, err1_eof := pass.Fetch(0)
			if err1_eof != nil {
				fmt.Println("PassThrough EOF fetch failed error : %v\n", err1_eof)
			}

			for i := 0; i < len(opkts_eof); i++ {
				_, err_enc_eof := encoder.Call([]*bmf.Packet{opkts_eof[i]})
				if err_enc_eof != nil {
					fmt.Printf("encoder EOF call failed! error : %v\n", err_enc_eof)
				}
			}
			break
		}

		if err != nil {
			fmt.Printf("decoder execute failed! error : %v\n", err)
			break
		}

		opkts_dec, err1 := decoder.Fetch(0)
		if err1 != nil {
			fmt.Printf("decoder fetch failed!\n, error : %v", err1)
			break
		}

		// do pass_through module
		for i := 0; i < len(opkts_dec); i++ {
			ipkts_pass := []*bmf.Packet{}
			ipkts_pass = append(ipkts_pass, opkts_dec[i])
			_, err2 := pass.Execute(ipkts_pass, true)
			if err2 != nil {
				fmt.Printf("Pass Through failed! %v\n", err2)
				break
			}

			opkts_pass, err1 := pass.Fetch(0)
			if err1 != nil {
				fmt.Printf("Pass Through Fetch failed! error : %v\n", err1)
				break
			}

			// do encoder module
			for i := 0; i < len(opkts_pass); i++ {
				_, err := encoder.Call([]*bmf.Packet{opkts_pass[i]})
				if err != nil {
					fmt.Printf("encoder call failed! error : %v\n", err)
				}
			}

		}
	}
}

func main() {
	start := time.Now()
	if os.Args[1] == "normalMode" {
		normalMode()
	}
	if os.Args[1] == "premoduleMode" {
		premoduleMode()
	}
	if os.Args[1] == "syncMode" {
		syncMode()
	}
	if os.Args[1] == "syncModeSerial" {
		syncModeSerial()
	}
	if os.Args[1] == "testAudio" {
		testAudio()
	}
	if os.Args[1] == "testWithInputOnlyAudio" {
		testWithInputOnlyAudio()
	}
	if os.Args[1] == "testWithEncodeWithAudioStreamButNoAudioFrame" {
		testWithEncodeWithAudioStreamButNoAudioFrame()
	}
	if os.Args[1] == "testWithNullAudio" {
		testWithNullAudio()
	}
	if os.Args[1] == "testSimple" {
		testSimple()
	}
	if os.Args[1] == "testhls" {
		testhls()
	}
	if os.Args[1] == "testCrypt" {
		testCrypt()
	}
	if os.Args[1] == "testOption" {
		testOption()
	}
	if os.Args[1] == "testImage" {
		testImage()
	}
	if os.Args[1] == "testVideo" {
		testVideo()
	}
	if os.Args[1] == "testConcatVideoAndAudio" {
		testConcatVideoAndAudio()
	}
	if os.Args[1] == "testShortVideoConcat" {
		testShortVideoConcat()
	}
	if os.Args[1] == "testMapParam" {
		testMapParam()
	}
	if os.Args[1] == "testRGB2Video" {
		testRGB2Video()
	}
	if os.Args[1] == "testStreamCopy" {
		testStreamCopy()
	}
	if os.Args[1] == "testStreamAudioCopy" {
		testStreamAudioCopy()
	}
	if os.Args[1] == "testExtractFrames" {
		testExtractFrames()
	}
	if os.Args[1] == "testIncorrectStreamNotify" {
		testIncorrectStreamNotify()
	}
	if os.Args[1] == "testIncorrectEncoderParam" {
		testIncorrectEncoderParam()
	}
	if os.Args[1] == "testDurations" {
		testDurations()
	}
	if os.Args[1] == "testOutputRawVideo" {
		testOutputRawVideo()
	}
	if os.Args[1] == "testOutputNull" {
		testOutputNull()
	}
	if os.Args[1] == "testVframes" {
		testVframes()
	}
	if os.Args[1] == "testSegmentTrans" {
		testSegmentTrans()
	}
	if os.Args[1] == "testEncoderPushOutputMp4" {
		testEncoderPushOutputMp4()
	}
	if os.Args[1] == "testEncoderPushOutputImage2Pipe" {
		testEncoderPushOutputImage2Pipe()
	}
	if os.Args[1] == "testEncoderPushOutputAudioPcmS16le" {
		testEncoderPushOutputAudioPcmS16le()
	}
	if os.Args[1] == "testSkipFrame" {
		testSkipFrame()
	}
	if os.Args[1] == "testAudioModule" {
		testAudioModule()
	}
	if os.Args[1] == "testExceptionInPythonModule" {
		testExceptionInPythonModule()
	}
	if os.Args[1] == "testVideoOverlays" {
		testVideoOverlays()
	}
	if os.Args[1] == "testVideoConcat" {
		testVideoConcat()
	}
	if os.Args[1] == "testSyncVideoFrame" {
		testSyncVideoFrame()
	}
	end := time.Since(start)
	fmt.Println("total process time is : ", end)
}
