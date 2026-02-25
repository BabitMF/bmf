import os
import sys
import time
import unittest
import subprocess as sp

if os.name == "nt":
    class timeout_decorator:
        @staticmethod
        def timeout(*args, **kwargs):
            return lambda f: f
else:
    import timeout_decorator

sys.path.append("../../test/")

from base_test.base_test_case import BaseTestCase
from base_test.media_info import MediaInfo

import bmf
import numpy as np
from bmf import bmf_sync, Packet, Task, ProcessResult
from decord import VideoReader, cpu

def bmf_sync_decode_packets(decode_param, stream_name="video"):
    decoder = bmf_sync.sync_module(
        "c_ffmpeg_decoder",
        decode_param,
        [],
        [0]
    )
    decoder.init()
    packets = []
    while True:
        task = Task(0, decoder.get_input_streams(), decoder.get_output_streams())
        result = decoder.process(task)
        q = task.get_outputs()[0]
        while not q.empty():
            pkt = q.get()
            if pkt.is_(bmf.VideoFrame):
                packets.append(pkt)
        if task.timestamp == bmf.Timestamp.DONE:
            break
    decoder.close()
    return packets

def bmf_decode_packets(decode_param, stream_name="video"):
    graph = bmf.graph()
    streams = graph.decode(decode_param)
    packet_generator = streams[stream_name].start()
    packets = []
    for pkt in packet_generator:
        if pkt.is_(bmf.VideoFrame):
            packets.append(pkt)
    return packets

def bmf_decode_timestamps(decode_param, stream_name="video", sync_module=True):
    packets = bmf_sync_decode_packets(decode_param, stream_name) if sync_module else bmf_decode_packets(decode_param, stream_name)
    return [pkt.timestamp for pkt in packets]

def bmf_decode_videoframes(decode_param, stream_name="video", sync_module=True):
    packets = bmf_sync_decode_packets(decode_param, stream_name) if sync_module else bmf_decode_packets(decode_param, stream_name)
    return [pkt.get(bmf.VideoFrame) for pkt in packets]

def bmf_decode_videoframes_with_metadata(decode_param, stream_name="video", sync_module=True):
    """Extract videoframes along with their dimensions and timestamps"""
    packets = bmf_sync_decode_packets(decode_param, stream_name) if sync_module else bmf_decode_packets(decode_param, stream_name)
    frames = []
    for pkt in packets:
        vf = pkt.get(bmf.VideoFrame)
        frames.append({
            'frame': vf,
            'width': vf.width,
            'height': vf.height,
            'timestamp': pkt.timestamp
        })
    return frames

TEST_FILES_DIR = "../../../output/files"

class TestDecordDecoder(BaseTestCase):
    # Performance data storage
    performance_data = []
    
    def setUp(self):
        super().setUp()
        ffpmeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")
        short_video_path = os.path.join(TEST_FILES_DIR, "big_bunny_10s_30fps.mp4")
        if not os.path.exists(short_video_path):
            raise FileNotFoundError(short_video_path)
        vfr_video_path = os.path.join(TEST_FILES_DIR, "big_bunny_10s_30fps_vfr.mp4")
        if not os.path.exists(vfr_video_path):
            sp.run([ffpmeg_path, "-i", short_video_path, "-vf", "mpdecimate", "-vsync", "vfr", vfr_video_path], check=True)

    def _get_video_paths(self):
        short_video = os.path.join(TEST_FILES_DIR, "big_bunny_10s_30fps.mp4")
        long_video = os.path.join(TEST_FILES_DIR, "big_bunny_1min_30fps.mp4")
        vfr_video = os.path.join(TEST_FILES_DIR, "big_bunny_10s_30fps_vfr.mp4")
        if not os.path.exists(short_video):
            raise FileNotFoundError(short_video)
        if not os.path.exists(long_video):
            raise FileNotFoundError(long_video)
        if not os.path.exists(vfr_video):
            raise FileNotFoundError(vfr_video)
        return {"10s": short_video, "1min": long_video, "10s_vfr": vfr_video}

    def _get_meta(self, video_path):
        mi = MediaInfo(video_path)
        duration = mi.get_duration()
        v_stream = mi.av_out_info.get('v_stream', {})
        n_total = int(v_stream.get('nb_frames', 0))
        avg_fps = mi.parse_fraction(v_stream.get('avg_frame_rate', '0/1'))
        if n_total == 0 and avg_fps > 0 and duration > 0:
            n_total = int(duration * avg_fps)
        return n_total, avg_fps, duration

    def _indices_for_fps(self, duration, n_total, fps):
        if duration <= 0 or n_total <= 0 or fps <= 0:
            return np.array([0], dtype=np.int64)
        count = int(duration * fps) + 1
        times = np.arange(count, dtype=np.float64) / fps
        indices = np.rint(times * (n_total / duration)).astype(np.int64)
        indices = indices[indices < n_total]
        return np.unique(indices)

    def _indices_for_n_frames(self, duration, n_total, n_frames):
        if n_frames <= 1 or n_total <= 0 or duration <= 0:
            return np.array([0], dtype=np.int64)
        times = np.linspace(0, duration, n_frames)
        indices = np.rint(times * (n_total - 1) / duration).astype(np.int64)
        return np.unique(indices)

    def _compare_bmf_vs_decord(self, video_path, mode, params):
        n_total, avg_fps, duration = self._get_meta(video_path)
        if mode == "fps":
            fps = params["fps"]
            indices = self._indices_for_fps(duration, n_total, fps).tolist()
            extract_params = {"fps": fps}
            tol_us = int(max((1.5 / max(fps, 1e-6)) * 1e6, 20000))
        elif mode == "n_frames":
            n = params["n_frames"]
            indices = self._indices_for_n_frames(duration, n_total, n).tolist()
            extract_params = {"n_frames": n}
            tol_us = int(max((1.5 / avg_fps) * 1e6, 20000))
        elif mode == "indices":
            indices = list(params["frame_indexes"])
            extract_params = {"frame_indexes": indices}
            tol_us = int(max((1.5 / avg_fps) * 1e6, 20000))
        else:
            raise ValueError("unknown mode")

        decord_start = time.perf_counter()
        vr = VideoReader(video_path, ctx=cpu(0))
        vr.get_batch(indices)
        decord_time = time.perf_counter() - decord_start
        ts_pair = vr.get_frame_timestamp(indices)
        decord_ts_us = (ts_pair[:, 0] * 1e6).astype(np.int64)

        bmf_start = time.perf_counter()
        bmf_ts_us = self._bmf_extract_timestamps_us(video_path, extract_params)
        bmf_time = time.perf_counter() - bmf_start

        if len(bmf_ts_us) != len(decord_ts_us):
            print(f"Length mismatch: BMF {len(bmf_ts_us)} vs Decord {len(decord_ts_us)}")
            print(f"BMF timestamps: {bmf_ts_us}")
            print(f"Decord timestamps: {decord_ts_us}")
        self.assertEqual(len(bmf_ts_us), len(decord_ts_us))
        diffs = np.abs(np.array(bmf_ts_us, dtype=np.int64) - decord_ts_us)
        if not np.all(diffs <= tol_us):
            max_diff = np.max(diffs)
            print(f"Timestamp mismatch: max diff {max_diff} > tol {tol_us}")
            print(f"BMF timestamps: {bmf_ts_us}")
            print(f"Decord timestamps: {decord_ts_us}")
            print(f"Diffs: {diffs}")
            mismatch_indices = np.where(diffs > tol_us)[0]
            print(f"Mismatch indices: {mismatch_indices}")
            for idx in mismatch_indices[:10]:  # Limit output to first 10 mismatches
                print(f"Index {idx}: BMF {bmf_ts_us[idx]} vs Decord {decord_ts_us[idx]} (diff {diffs[idx]})")
        self.assertTrue(np.all(diffs <= tol_us))
        self.assertTrue(decord_time > 0)
        self.assertTrue(bmf_time > 0)
        
        # Store performance data
        performance_ratio = bmf_time / max(decord_time, 1e-6)
        TestDecordDecoder.performance_data.append({
            "test_type": "sampling_comparison",
            "video": os.path.basename(video_path),
            "mode": mode,
            "params": str(extract_params),
            "bmf_time_s": bmf_time,
            "decord_time_s": decord_time,
            "ratio": performance_ratio,
            "frames_extracted": len(bmf_ts_us)
        })
        
        print(
            "video",
            os.path.basename(video_path),
            "mode",
            mode,
            "params",
            extract_params,
            "bmf_s",
            bmf_time,
            "decord_s",
            decord_time,
            "ratio",
            performance_ratio,
        )

    def _bmf_extract_timestamps_us(self, video_path, extract_params):
        video_param = {
            "input_path": video_path,
            "video_params": {"extract_frames": extract_params},
        }
        return bmf_decode_timestamps(video_param)

    @timeout_decorator.timeout(seconds=240)
    def test_fps_sampling_matches_decord(self):
        for name, path in self._get_video_paths().items():
            for fps in [0.25, 0.5, 1.0, 2.0]:
                if "vfr" in name: 
                    continue # the fps test is not applicable for vfr videos
                with self.subTest(video=name, fps=fps):
                    self._compare_bmf_vs_decord(path, "fps", {"fps": fps})

    @timeout_decorator.timeout(seconds=240)
    def test_n_frames_sampling_matches_decord(self):
        for name, path in self._get_video_paths().items():
            if "vfr" in name: 
                continue # the n_frames test is not applicable for vfr videos
            for n in [1, 2, 5, 10, 30, 100]:
                with self.subTest(video=name, n_frames=n):
                    self._compare_bmf_vs_decord(path, "n_frames", {"n_frames": n})

    @timeout_decorator.timeout(seconds=240)
    def test_index_sampling_matches_decord(self):
        for name, path in self._get_video_paths().items():
            vr = VideoReader(path, ctx=cpu(0))
            indices = [0, 5, 10, min(20, len(vr) - 1), len(vr) - 1]
            self._compare_bmf_vs_decord(path, "indices", {"frame_indexes": indices})

    @timeout_decorator.timeout(seconds=240)
    def test_naive_vs_sampling_performance(self):
        video_paths = self._get_video_paths()
        long_video_path = video_paths["1min"]
        n_total, avg_fps, duration = self._get_meta(long_video_path)
        naive_start = time.perf_counter()
        naive_timestamps = bmf_decode_timestamps({"input_path": long_video_path})
        naive_time = time.perf_counter() - naive_start
        print(f"Naive: {len(naive_timestamps)} frames, {naive_time:.3f}s")
        
        # Store naive performance data
        TestDecordDecoder.performance_data.append({
            "test_type": "naive_vs_sampling",
            "video": os.path.basename(long_video_path),
            "mode": "naive",
            "params": "full_decode",
            "bmf_time_s": naive_time,
            "decord_time_s": 0.0,
            "ratio": 0.0,
            "frames_extracted": len(naive_timestamps),
            "speedup": 1.0
        })
        
        if os.getenv("PERF_COMPREHENSIVE", "false").lower() == "true":
            n_frames_opts = [1, 5, 10, 20, 30, 60, 120]
        else:
            n_frames_opts = [1, 30, 120]

        for n_frames in n_frames_opts:
            if n_frames > n_total:
                continue
            with self.subTest(n_frames=n_frames):
                sampling_start = time.perf_counter()
                sampling_param = {
                    "input_path": long_video_path,
                    "video_params": {"extract_frames": {"n_frames": n_frames}}
                }
                sampling_timestamps = bmf_decode_timestamps(sampling_param)
                sampling_time = time.perf_counter() - sampling_start
                speedup = naive_time / max(sampling_time, 1e-6)
                print(f"Sample {n_frames}: {sampling_time:.3f}s, {speedup:.2f}x")
                
                # Store sampling performance data
                TestDecordDecoder.performance_data.append({
                    "test_type": "naive_vs_sampling",
                    "video": os.path.basename(long_video_path),
                    "mode": "n_frames",
                    "params": f"n_frames={n_frames}",
                    "bmf_time_s": sampling_time,
                    "decord_time_s": 0.0,
                    "ratio": 0.0,
                    "frames_extracted": n_frames,
                    "speedup": speedup
                })
                
                self.assertEqual(len(sampling_timestamps), n_frames)
                if n_frames < n_total * 0.5:
                    self.assertLess(sampling_time, naive_time)

    @timeout_decorator.timeout(seconds=240)
    def test_ffmpeg_filter_param_fps(self):
        video_paths = self._get_video_paths()
        short_video_path = video_paths["10s"]
        for fps in [2, 5, 10]:
            with self.subTest(video="10s", fps=fps):
                decode_param = {
                    "input_path": short_video_path,
                    "filter_param": {"fps": fps}
                }
                # Test timestamps
                timestamps = bmf_decode_timestamps(decode_param)
                n_total, avg_fps, duration = self._get_meta(short_video_path)
                expected_frames = int(duration * fps)
                self.assertGreater(len(timestamps), 0)
                self.assertLessEqual(len(timestamps), expected_frames + 2)
                
                # Test frame properties
                frames = bmf_decode_videoframes_with_metadata(decode_param)
                self.assertEqual(len(frames), len(timestamps))
                
                # Verify frame dimensions match original video
                original_vr = VideoReader(short_video_path, ctx=cpu(0))
                original_width, original_height = original_vr[0].shape[1], original_vr[0].shape[0]
                
                for frame_info in frames:
                    self.assertEqual(frame_info['width'], original_width)
                    self.assertEqual(frame_info['height'], original_height)
                    self.assertIsNotNone(frame_info['frame'])
                    self.assertIsInstance(frame_info['timestamp'], (int, float))
                
                print(f"FFmpeg FPS {fps}: {len(timestamps)} frames, dimensions verified")

    @timeout_decorator.timeout(seconds=240)
    def test_ffmpeg_filter_param_crop(self):
        video_paths = self._get_video_paths()
        for name, path in video_paths.items():
            crop_params = {
                "x": 100,
                "y": 100,
                "w": 640,
                "h": 480
            }
            with self.subTest(video=name):
                decode_param = {
                    "input_path": path,
                    "filter_param": {"crop": crop_params}
                }
                # Test timestamps
                timestamps = bmf_decode_timestamps(decode_param)
                self.assertGreater(len(timestamps), 0)
                
                # Test frame properties with crop verification
                frames = bmf_decode_videoframes_with_metadata(decode_param)
                self.assertEqual(len(frames), len(timestamps))
                
                # Verify crop dimensions
                for frame_info in frames:
                    self.assertEqual(frame_info['width'], crop_params['w'])
                    self.assertEqual(frame_info['height'], crop_params['h'])
                    self.assertIsNotNone(frame_info['frame'])
                    self.assertIsInstance(frame_info['timestamp'], (int, float))
                
                print(f"FFmpeg Crop: {len(timestamps)} frames from {name}, crop dimensions {crop_params['w']}x{crop_params['h']} verified")

    @timeout_decorator.timeout(seconds=240)
    def test_ffmpeg_filter_param_scale(self):
        video_paths = self._get_video_paths()
        for name, path in video_paths.items():
            scale_params = {
                "w": 640,
                "h": 480
            }
            with self.subTest(video=name):
                decode_param = {
                    "input_path": path,
                    "filter_param": {"scale": scale_params}
                }
                # Test timestamps
                timestamps = bmf_decode_timestamps(decode_param)
                self.assertGreater(len(timestamps), 0)
                
                # Test frame properties with scale verification
                frames = bmf_decode_videoframes_with_metadata(decode_param)
                self.assertEqual(len(frames), len(timestamps))
                
                # Verify scale dimensions
                for frame_info in frames:
                    self.assertEqual(frame_info['width'], scale_params['w'])
                    self.assertEqual(frame_info['height'], scale_params['h'])
                    self.assertIsNotNone(frame_info['frame'])
                    self.assertIsInstance(frame_info['timestamp'], (int, float))
                
                print(f"FFmpeg Scale: {len(timestamps)} frames from {name}, scale dimensions {scale_params['w']}x{scale_params['h']} verified")

    @timeout_decorator.timeout(seconds=240)
    def test_ffmpeg_filter_param_combined(self):
        video_paths = self._get_video_paths()
        for name, path in video_paths.items():
            with self.subTest(video=name):
                decode_param = {
                    "input_path": path,
                    "filter_param": {
                        "fps": 10,
                        "crop": {"x": 100, "y": 100, "w": 640, "h": 480},
                        "scale": {"w": 320, "h": 240}
                    }
                }
                # Test timestamps
                timestamps = bmf_decode_timestamps(decode_param)
                self.assertGreater(len(timestamps), 0)
                n_total, avg_fps, duration = self._get_meta(path)
                expected_frames = int(duration * 10)
                self.assertLessEqual(len(timestamps), expected_frames + 2)
                
                # Test frame properties with combined filter verification
                frames = bmf_decode_videoframes_with_metadata(decode_param)
                self.assertEqual(len(frames), len(timestamps))
                
                # Verify combined filter dimensions (scale should be applied after crop)
                for frame_info in frames:
                    self.assertEqual(frame_info['width'], 320)  # Final scale width
                    self.assertEqual(frame_info['height'], 240)  # Final scale height
                    self.assertIsNotNone(frame_info['frame'])
                    self.assertIsInstance(frame_info['timestamp'], (int, float))
                
                print(f"FFmpeg Combined: {len(timestamps)} frames from {name}, final dimensions 320x240 verified")

    @classmethod
    def tearDownClass(cls):
        """Print performance data as ASCII table after all tests complete"""
        if cls.performance_data:
            print("\n" + "="*80)
            print("PERFORMANCE SUMMARY")
            print("="*80)
            
            # Group data by test type
            grouped_data = {}
            for entry in cls.performance_data:
                test_type = entry["test_type"]
                if test_type not in grouped_data:
                    grouped_data[test_type] = []
                grouped_data[test_type].append(entry)
            
            # Print sampling comparison table
            if "sampling_comparison" in grouped_data:
                print("\nSampling Comparison (BMF vs Decord):")
                print("| Video | Mode | Params | Frames | BMF Time (s) | Decord Time (s) | Ratio |")
                print("|-------|------|--------|--------|--------------|-----------------|-------|")
                for entry in grouped_data["sampling_comparison"]:
                    video = entry["video"]
                    mode = entry["mode"]
                    params = entry["params"][:30]  # Truncate long params
                    frames = entry["frames_extracted"]
                    bmf_time = f"{entry['bmf_time_s']:.3f}"
                    decord_time = f"{entry['decord_time_s']:.3f}"
                    ratio = f"{entry['ratio']:.2f}"
                    print(f"| {video} | {mode} | {params} | {frames} | {bmf_time} | {decord_time} | {ratio} |")
            
            # Print naive vs sampling performance table
            if "naive_vs_sampling" in grouped_data:
                print("\nNaive vs N-Frames Seek-Based Sampling Performance:")
                print("| Video | Mode | Params | Frames | BMF Time (s) | Speedup |")
                print("|-------|------|--------|--------|--------------|---------|")
                for entry in grouped_data["naive_vs_sampling"]:
                    video = entry["video"]
                    mode = entry["mode"]
                    params = entry["params"][:30]  # Truncate long params
                    frames = entry["frames_extracted"]
                    bmf_time = f"{entry['bmf_time_s']:.3f}"
                    speedup = f"{entry.get('speedup', 0):.2f}x" if entry.get('speedup') else "N/A"
                    print(f"| {video} | {mode} | {params} | {frames} | {bmf_time} | {speedup} |")

if __name__ == "__main__":
    unittest.main()