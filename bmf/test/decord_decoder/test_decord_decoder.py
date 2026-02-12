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

class TestDecordDecoder(BaseTestCase):
    def _get_video_paths(self):
        files_dir = "../../../output/files"
        short_video = os.path.join(files_dir, "big_bunny_10s_30fps.mp4")
        long_video = os.path.join(files_dir, "big_bunny_1min_30fps.mp4")
        if not os.path.exists(short_video):
            raise FileNotFoundError(short_video)
        if not os.path.exists(long_video):
            raise FileNotFoundError(long_video)
        return {
            "10s": short_video,
            "1min": long_video
        }

    def _bmf_extract_timestamps_us(self, video_path, extract_params):
        graph = bmf.graph()
        streams = graph.decode({
            "input_path": video_path,
            "video_params": {
                "extract_frames": extract_params
            }
        })
        pkts = streams["video"].start()
        timestamps = []
        for pkt in pkts:
            if pkt.is_(bmf.VideoFrame):
                timestamps.append(int(pkt.timestamp))
            else:
                break
        return timestamps

    def _bmf_decode_all_timestamps_us(self, video_path):
        graph = bmf.graph()
        streams = graph.decode({
            "input_path": video_path
        })
        pkts = streams["video"].start()
        timestamps = []
        for pkt in pkts:
            if pkt.is_(bmf.VideoFrame):
                timestamps.append(int(pkt.timestamp))
            else:
                break
        return timestamps

    def _decord_frames(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0))
        frame_ts = vr.get_frame_timestamp(list(range(len(vr))))
        start_ts = frame_ts[:, 0]
        end_ts = frame_ts[:, 1]
        return vr, start_ts, end_ts

    def _expected_times_fps(self, start_ts, end_ts, fps):
        duration = end_ts[-1] - start_ts[0]
        if duration <= 0:
            return np.array([start_ts[0]])
        step = 1.0 / fps
        count = int(duration / step)
        return start_ts[0] + np.arange(count + 1) * step

    def _expected_times_n_frames(self, start_ts, end_ts, n_frames):
        duration = end_ts[-1] - start_ts[0]
        if n_frames <= 1 or duration <= 0:
            return np.array([start_ts[0]])
        step = duration / (n_frames - 1)
        return start_ts[0] + np.arange(n_frames) * step

    def _expected_times_indexes(self, start_ts, indices):
        return start_ts[np.array(indices, dtype=np.int64)]

    def _match_times_to_indices(self, start_ts, target_times):
        target_times = np.array(target_times)
        indices = np.abs(start_ts[:, None] - target_times[None, :]).argmin(axis=0)
        return indices

    def _compare_alignment(self, bmf_ts_us, expected_times_s, avg_fps):
        expected_us = (expected_times_s * 1e6).astype(np.int64)
        self.assertEqual(len(bmf_ts_us), len(expected_us))
        tolerance_us = int(max((1.5 / avg_fps) * 1e6, 20000))
        diffs = np.abs(np.array(bmf_ts_us) - expected_us)
        self.assertTrue(np.all(diffs <= tolerance_us))

    def _compare_alignment_unordered(self, bmf_ts_us, expected_times_s, avg_fps):
        expected_us = (expected_times_s * 1e6).astype(np.int64)
        self.assertEqual(len(bmf_ts_us), len(expected_us))
        tolerance_us = int(max((1.5 / avg_fps) * 1e6, 20000))
        diffs = np.abs(np.sort(np.array(bmf_ts_us)) - np.sort(expected_us))
        self.assertTrue(np.all(diffs <= tolerance_us))

    def _run_case(self, video_path, extract_params, expected_times_fn):
        vr, start_ts, end_ts = self._decord_frames(video_path)
        avg_fps = vr.get_avg_fps()
        expected_times = expected_times_fn(start_ts, end_ts)

        decord_indices = self._match_times_to_indices(start_ts, expected_times)
        decord_start = time.perf_counter()
        vr.get_batch(decord_indices.tolist())
        decord_time = time.perf_counter() - decord_start

        bmf_start = time.perf_counter()
        bmf_ts_us = self._bmf_extract_timestamps_us(video_path, extract_params)
        bmf_time = time.perf_counter() - bmf_start

        self._compare_alignment(bmf_ts_us, expected_times, avg_fps)
        self.assertTrue(decord_time > 0)
        self.assertTrue(bmf_time > 0)
        print(
            "video",
            os.path.basename(video_path),
            "params",
            extract_params,
            "bmf_s",
            bmf_time,
            "decord_s",
            decord_time,
            "ratio",
            bmf_time / max(decord_time, 1e-6),
        )

    @timeout_decorator.timeout(seconds=240)
    def test_extract_frames_fps(self):
        for name, path in self._get_video_paths().items():
            self._run_case(
                path,
                {"fps": 0.5},
                lambda start_ts, end_ts: self._expected_times_fps(start_ts, end_ts, 0.5),
            )

    @timeout_decorator.timeout(seconds=240)
    def test_extract_frames_n_frames(self):
        for name, path in self._get_video_paths().items():
            self._run_case(
                path,
                {"n_frames": 7},
                lambda start_ts, end_ts: self._expected_times_n_frames(start_ts, end_ts, 7),
            )

    @timeout_decorator.timeout(seconds=240)
    def test_extract_frames_frame_indexes(self):
        for name, path in self._get_video_paths().items():
            vr, start_ts, end_ts = self._decord_frames(path)
            indices = [0, 5, 10, 20, len(vr) - 1]
            expected_times = self._expected_times_indexes(start_ts, indices)

            decord_start = time.perf_counter()
            vr.get_batch(indices)
            decord_time = time.perf_counter() - decord_start

            bmf_start = time.perf_counter()
            bmf_ts_us = self._bmf_extract_timestamps_us(
                path,
                {"frame_indexes": indices},
            )
            bmf_time = time.perf_counter() - bmf_start

            avg_fps = vr.get_avg_fps()
            self._compare_alignment(bmf_ts_us, expected_times, avg_fps)
            self.assertTrue(decord_time > 0)
            self.assertTrue(bmf_time > 0)
            print(
                "video",
                os.path.basename(path),
                "params",
                {"frame_indexes": indices},
                "bmf_s",
                bmf_time,
                "decord_s",
                decord_time,
                "ratio",
                bmf_time / max(decord_time, 1e-6),
            )

    @timeout_decorator.timeout(seconds=240)
    def test_extract_frames_frame_indexes_duplicates(self):
        path = self._get_video_paths()["10s"]
        vr, start_ts, end_ts = self._decord_frames(path)
        max_index = len(vr) - 1
        indices = [0, 5, 5, 10, 10, 0, max_index, max_index]
        expected_times = self._expected_times_indexes(start_ts, indices)

        bmf_ts_us = self._bmf_extract_timestamps_us(
            path,
            {"frame_indexes": indices},
        )

        avg_fps = vr.get_avg_fps()
        self._compare_alignment_unordered(bmf_ts_us, expected_times, avg_fps)

    @timeout_decorator.timeout(seconds=240)
    def test_extract_frames_frame_indexes_non_increasing(self):
        path = self._get_video_paths()["10s"]
        vr, start_ts, end_ts = self._decord_frames(path)
        max_index = len(vr) - 1
        indices = [min(max_index, i) for i in [20, 10, 10, 5, 2, 0]]
        expected_times = self._expected_times_indexes(start_ts, indices)

        bmf_ts_us = self._bmf_extract_timestamps_us(
            path,
            {"frame_indexes": indices},
        )

        avg_fps = vr.get_avg_fps()
        self._compare_alignment_unordered(bmf_ts_us, expected_times, avg_fps)

    @timeout_decorator.timeout(seconds=240)
    def test_extract_frames_n_frames_perf_vs_naive(self):
        path = self._get_video_paths()["1min"]
        vr, start_ts, end_ts = self._decord_frames(path)
        avg_fps = vr.get_avg_fps()
        n_frames = 7
        expected_times = self._expected_times_n_frames(start_ts, end_ts, n_frames)

        naive_start = time.perf_counter()
        all_ts_us = self._bmf_decode_all_timestamps_us(path)
        self.assertTrue(len(all_ts_us) > 0)
        indices = np.linspace(0, len(all_ts_us) - 1, n_frames).astype(np.int64)
        naive_sampled_ts_us = [all_ts_us[i] for i in indices]
        naive_time = time.perf_counter() - naive_start

        bmf_start = time.perf_counter()
        bmf_ts_us = self._bmf_extract_timestamps_us(
            path,
            {"n_frames": n_frames},
        )
        bmf_time = time.perf_counter() - bmf_start

        self._compare_alignment(naive_sampled_ts_us, expected_times, avg_fps)
        self._compare_alignment(bmf_ts_us, expected_times, avg_fps)
        self.assertTrue(naive_time > 0)
        self.assertTrue(bmf_time > 0)
        print(
            "video",
            os.path.basename(path),
            "params",
            {"naive_n_frames": n_frames, "bmf_n_frames": n_frames},
            "naive_s",
            naive_time,
            "bmf_s",
            bmf_time,
            "ratio",
            naive_time / max(bmf_time, 1e-6),
        )

if __name__ == "__main__":
    unittest.main()
