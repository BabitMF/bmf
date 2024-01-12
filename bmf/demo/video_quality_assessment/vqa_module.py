#!/usr/bin/env python
# -*- coding: utf-8 -*-


from module_utils.template import SyncModule
from module_utils.logger import get_logger
import os
import time
import json
import os.path as osp
import numpy as np
import onnxruntime as ort

#设置torch占用的内核数
os.environ["OMP_NUM_THREADS"] = "8"
LOGGER = get_logger()



def random_crop(d_img, crop_size):
    b, c, h, w = d_img.shape
    top = np.random.randint(0, h - crop_size)
    left = np.random.randint(0, w - crop_size)
    crop_img = d_img[:, :, top:top + crop_size, left:left + crop_size]
    return crop_img

def crop_for_video(img, crop_size, num_crop):
    for i in range(num_crop):
        if i==0:
            crop_img=random_crop(img,crop_size)
        else:
            crop_img=np.concatenate(( crop_img,random_crop(img,crop_size) ), axis=0)
    return crop_img




class VQA_4kpgc:
    def __init__(self, output_path, model_version=1, width=224, height=224, num_crop=5, caiyangjiange=100):
        self._frm_idx = 0
        self._frm_scores = []
        self._output_path = output_path
        self._model_version = model_version

        self.num_crop=num_crop
        self.resize_reso = [width, height]
        self.caiyangjiange = caiyangjiange
    
        model_dir = osp.join(osp.abspath(osp.dirname(__file__)), 'models')
        vqa_4kpgc_model_path = osp.realpath(osp.join(model_dir, 'vqa_4kpgc_1.onnx'))
        self.ort_session = ort.InferenceSession(vqa_4kpgc_model_path)
        self.input_node=self.ort_session.get_inputs()[0]

        LOGGER.info("create AdvColor model [CPU]")


    def preprocess(self, frame):
        frame = (frame.astype(np.float32) / 255.0 - np.array([0.5, 0.5, 0.5], dtype='float32')) / \
            (np.array([0.5, 0.5, 0.5], dtype='float32'))
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, 0)
        frame = crop_for_video(frame, self.resize_reso[0], self.num_crop)
        return frame


    @staticmethod
    def score_pred_mapping(preds):
        max=9.8
        min=0.111111111111111
        pred_score=preds*(max-min)+min
        return pred_score


    def process(self, frames):
        self._frm_idx += 1
        #对同一视频间隔一定帧计算一次 OR 采用module_utils中的segment_decode_ticks函数进行解码
        #if (self._frm_idx-1)%self.caiyangjiange==0:
        frames = [frame if frame.flags["C_CONTIGUOUS"] else np.ascontiguousarray(frame) for frame in frames]
        frame = self.preprocess(frames[0])
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame, dtype=np.float32)

        t1 = time.time()
        raw_score = self.ort_session.run(None, {self.input_node.name: frame})[0].mean()
        score = self.score_pred_mapping(raw_score)
        self._frm_scores.append(score)
        t2 = time.time()
        LOGGER.info(f'[vqa_4kpgc] inference time: {(t2 - t1)*1000:0.1f} ms')
        
        return frames[0]


    def clean(self):
        nr_score = round(np.mean(self._frm_scores), 2)
        results = {'vqa_4kpgc': nr_score, 'vqa_4kpgc_version': self._model_version, 'num_crop': self.num_crop, 'cau_frames_num': self._frm_idx}
        LOGGER.info(f'overall prediction {json.dumps(results)}')
        with open(self._output_path, 'w') as outfile:
            json.dump(results, outfile, indent=4, ensure_ascii=False)
        




class BMFVQA_4kpgc(SyncModule):
    def __init__(self, node=None, option=None):
        height = option.get('height', 0)
        width = option.get('width', 0)
        output_path = option.get('output_path', 0)
        model_version = option.get('model_version', 'v1.0')
        self._nrp = VQA_4kpgc(output_path=output_path, model_version=model_version, width=224, height=224, num_crop=5, caiyangjiange=100)
        SyncModule.__init__(self, node, nb_in=1, in_fmt='rgb24', out_fmt='rgb24')

    def core_process(self, frames):
        return self._nrp.process(frames)

    def clean(self):
        self._nrp.clean()



