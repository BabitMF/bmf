#!/usr/bin/env python
# -*- coding: utf-8 -*-

from module_utils import SyncModule
import os
import time
import json
import pdb
import os.path as osp
import numpy as np

os.environ["OMP_NUM_THREADS"] = "8"
import onnxruntime as ort
import torch
import logging
import cv2


def get_logger():
    return logging.getLogger("main")


LOGGER = get_logger()


def flex_resize_aesv2(img, desired_size=[448, 672], pad_color=[0, 0, 0]):
    old_h, old_w = img.shape[:2]  # old_size is in (height, width) format
    if desired_size[0] >= desired_size[1]:
        if old_h < old_w:  # rotate the honrizontal video
            img = np.rot90(img, k=1, axes=(1, 0))
    else:
        if old_h > old_w:  # rotate the vertical video
            img = np.rot90(img, k=1, axes=(1, 0))
    old_h, old_w = img.shape[:2]

    if old_w / old_h > (desired_size[1] / desired_size[0]):
        ratio = desired_size[0] / old_h
    else:
        ratio = desired_size[1] / old_w
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    h, w, _ = img.shape
    h_crop = (h - desired_size[0]) // 2
    w_crop = (w - desired_size[1]) // 2
    img = img[h_crop:h_crop + desired_size[0],
              w_crop:w_crop + desired_size[1], :]
    return img


class Aesmod:

    def __init__(self, model_path, model_version, output_path):
        self._frm_idx = 0
        self._frm_scores = []
        self._model_version = model_version
        self._output_path = output_path

        # model_dir = osp.join(osp.abspath(osp.dirname("__file__")), "models")
        # aesmod_ort_model_path = osp.realpath(
        #    osp.join(model_dir, "aes_transonnx_update3.onnx")
        # )
        self.use_gpu = False
        aesmod_ort_model_path = model_path
        print(aesmod_ort_model_path)
        LOGGER.info("loading aesthetic ort inference session")
        self.ort_session = ort.InferenceSession(aesmod_ort_model_path)

        self.resize_reso = [672, 448]

    def preprocess(self, frame):
        frame = flex_resize_aesv2(frame)
        # print('using flex_resize_aesv2', frame.shape)
        frame = (frame.astype(np.float32) / 255.0 -
                 np.array([0.485, 0.456, 0.406], dtype="float32")) / (np.array(
                     [0.229, 0.224, 0.225], dtype="float32"))
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, 0)
        return frame

    @staticmethod
    def tensor_to_list(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().flatten().tolist()
        else:
            return tensor.cpu().flatten().tolist()

    @staticmethod
    def score_pred_mapping(raw_scores, raw_min=2.60, raw_max=7.42):
        pred_score = np.clip(
            np.sum([x * (i + 1) for i, x in enumerate(raw_scores)]), raw_min,
            raw_max)
        pred_score = np.sqrt((pred_score - raw_min) / (raw_max - raw_min)) * 100
        return float(np.clip(pred_score, 0, 100.0))

    def process(self, frames):
        frames = [
            frame
            if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame)
            for frame in frames
        ]
        frame = self.preprocess(frames[0])
        print("after preprocess shape", frame.shape)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame, dtype=np.float32)

        t1 = time.time()
        if self.use_gpu:
            with torch.no_grad():
                input_batch = torch.from_numpy(frame).contiguous().cuda()
                preds, _ = self.trt_model(input_batch)
                raw_score = self.tensor_to_list(preds)
        else:

            raw_score = self.ort_session.run(None, {"input": frame})
            raw_score = raw_score[0][0]
        score = self.score_pred_mapping(raw_score)
        self._frm_scores.append(score)
        self._frm_idx += 1
        t2 = time.time()
        LOGGER.info(f"[Aesmod] inference time: {(t2 - t1) * 1000:0.1f} ms")
        return frames[0]

    def clean(self):
        nr_score = round(np.mean(self._frm_scores), 2)
        results = {
            "aesthetic": nr_score,
            "aesthetic_version": self._model_version
        }
        LOGGER.info(f"overall prediction {json.dumps(results)}")
        with open(self._output_path, "w") as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)


class BMFAesmod(SyncModule):

    def __init__(self, node=None, option=None):
        output_path = option.get("output_path", 0)
        model_version = option.get("model_version", "v1.0")
        model_path = option.get("model_path",
                                "../../models/aes_transonnx_update3.onnx")
        self._nrp = Aesmod(model_path, model_version, output_path)
        SyncModule.__init__(self,
                            node,
                            nb_in=1,
                            in_fmt="rgb24",
                            out_fmt="rgb24")

    def core_process(self, frames):
        return self._nrp.process(frames)

    def clean(self):
        self._nrp.clean()
