import os
import sys
import time
import unittest

if os.name == "nt":
    class timeout_decorator:
        @staticmethod
        def timeout(*args, **kwargs):
            return lambda f: f
else:
    import timeout_decorator

sys.path.append("../../test/")

from base_test.base_test_case import BaseTestCase

import bmf
import numpy as np
from decord import VideoReader, cpu

def bmf_decode(decode_param):
    graph = bmf.graph()
    streams = graph.decode(decode_param)
    video_pkts = streams["video"].start()
    timestamps = []
    for pkt in video_pkts:
        if pkt.is_(bmf.VideoFrame):
            timestamps.append(int(pkt.timestamp))
    return timestamps

class TestDecordDecoder(BaseTestCase):
    def _get_video_paths(self):
        files_dir = "../../../output/files"
        short_video = os.path.join(files_dir, "big_bunny_10s_30fps.mp4")
        long_video = os.path.join(files_dir, "big_bunny_1min_30fps.mp4")
        if not os.path.exists(short_video):
            raise FileNotFoundError(short_video)
        if not os.path.exists(long_video):
            raise FileNotFoundError(long_video)
        return {"10s": short_video, "1min": long_video}

    def _get_vr_meta(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        n_total = len(vr)
        avg_fps = max(vr.get_avg_fps(), 1e-6)
        duration = n_total / avg_fps if avg_fps > 0 else 0.0
        return vr, n_total, avg_fps, duration

    def _indices_for_fps(self, n_total, avg_fps, fps):
        if n_total <= 0 or avg_fps <= 0 or fps <= 0:
            return np.array([0], dtype=np.int64)
        duration = n_total / avg_fps
        count = int(duration * fps) + 1
        times = np.arange(count, dtype=np.float64) / fps
        indices = np.rint(times * avg_fps).astype(np.int64)
        indices = indices[indices < n_total]
        return np.unique(indices)

    def _indices_for_n_frames(self, n_total, n_frames):
        if n_frames <= 1 or n_total <= 0:
            return np.array([0], dtype=np.int64)
        indices = np.linspace(0, max(n_total - 1, 0), n_frames).astype(np.int64)
        return np.unique(indices)

    def _compare_bmf_vs_decord(self, video_path, mode, params):
        vr, n_total, avg_fps, duration = self._get_vr_meta(video_path)
        if mode == "fps":
            fps = params["fps"]
            indices = self._indices_for_fps(n_total, avg_fps, fps).tolist()
            extract_params = {"fps": fps}
            tol_us = int(max((1.5 / max(fps, 1e-6)) * 1e6, 20000))
        elif mode == "n_frames":
            n = params["n_frames"]
            indices = self._indices_for_n_frames(n_total, n).tolist()
            extract_params = {"n_frames": n}
            tol_us = int(max((1.5 / avg_fps) * 1e6, 20000))
        elif mode == "indices":
            indices = list(params["frame_indexes"])
            extract_params = {"frame_indexes": indices}
            tol_us = int(max((1.5 / avg_fps) * 1e6, 20000))
        else:
            raise ValueError("unknown mode")

        decord_start = time.perf_counter()
        vr.get_batch(indices)
        decord_time = time.perf_counter() - decord_start
        ts_pair = vr.get_frame_timestamp(indices)
        decord_ts_us = (ts_pair[:, 0] * 1e6).astype(np.int64)

        bmf_start = time.perf_counter()
        bmf_ts_us = self._bmf_extract_timestamps_us(video_path, extract_params)
        bmf_time = time.perf_counter() - bmf_start

        self.assertEqual(len(bmf_ts_us), len(decord_ts_us))
        diffs = np.abs(np.array(bmf_ts_us, dtype=np.int64) - decord_ts_us)
        self.assertTrue(np.all(diffs <= tol_us))
        self.assertTrue(decord_time > 0)
        self.assertTrue(bmf_time > 0)
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
            bmf_time / max(decord_time, 1e-6),
        )

    def _bmf_extract_timestamps_us(self, video_path, extract_params):
        video_param = {
            "input_path": video_path,
            "video_params": {"extract_frames": extract_params},
        }
        return bmf_decode(video_param)

    @timeout_decorator.timeout(seconds=240)
    def test_fps_sampling_matches_decord(self):
        for name, path in self._get_video_paths().items():
            for fps in [0.25, 0.5, 1.0, 2.0]:
                with self.subTest(video=name, fps=fps):
                    self._compare_bmf_vs_decord(path, "fps", {"fps": fps})

    @timeout_decorator.timeout(seconds=240)
    def test_n_frames_sampling_matches_decord(self):
        for name, path in self._get_video_paths().items():
            for n in [1, 2, 5, 10, 30, 100]:
                with self.subTest(video=name, n_frames=n):
                    self._compare_bmf_vs_decord(path, "n_frames", {"n_frames": n})

    @timeout_decorator.timeout(seconds=240)
    def test_index_sampling_matches_decord(self):
        for name, path in self._get_video_paths().items():
            vr = VideoReader(path, ctx=cpu(0))
            indices = [0, 5, 10, min(20, len(vr) - 1), len(vr) - 1]
            self._compare_bmf_vs_decord(path, "indices", {"frame_indexes": indices})

if __name__ == "__main__":
    unittest.main()
