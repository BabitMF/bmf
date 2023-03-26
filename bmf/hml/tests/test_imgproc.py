

import pytest
import numpy as np
import os
import cv2
from hml_fixtures import device_type, img_dtype, channel_format, channels
from hml_fixtures import mp, lenna_image, to_np_dtype
import glob


def cv2_resize_interp(mode):
    mmap = {
        mp.kNearest: cv2.INTER_NEAREST,
        mp.kBilinear : cv2.INTER_LINEAR,
        mp.kBicubic  : cv2.INTER_CUBIC
    }
    return mmap[mode]


def pre_process_image(im, device_type, format, img_dtype, channels=3):
    assert(channels <= im.shape[-1])
    im = im[..., :channels]

    if img_dtype == mp.kHalf:
        origin = im.astype(np.float32)
    else:
        origin = im.astype(to_np_dtype(img_dtype))

    mp_origin = mp.from_numpy(origin).to(device_type).unsqueeze(0)
    if img_dtype == mp.kHalf:
        mp_origin = mp_origin.to(mp.kHalf)

    if format == mp.kNCHW:
        mp_origin = mp_origin.permute((0, 3, 1, 2)).contiguous()

    return origin, mp_origin


class TestImageResize(object):
    @pytest.fixture(params=[pytest.param(m, id=repr(m)) for m in (mp.kNearest, mp.kBilinear, mp.kBicubic)])
    def mode(self, request):
        return request.param


    @pytest.fixture()
    def with_half(self):
        return True


    def test_image_resize(self, lenna_image, mode, device_type, img_dtype, channel_format):
        format = channel_format

        origin, mp_origin = pre_process_image(lenna_image, device_type, channel_format, img_dtype)

        sizes = [(240, 160), (320, 240), (480, 360), (1280, 720), (1920, 1080), (3840, 2160)]

        for size in sizes:
            ref = cv2.resize(origin, size, interpolation=cv2_resize_interp(mode))
            mp_out = mp.img.resize(mp_origin, size[0], size[1], mode=mode, format=format)
            if format == mp.kNCHW:
                mp_out = mp_out.permute((0, 2, 3, 1)) # convert to NHWC
            out = mp_out.cpu().numpy()

            df = np.abs(ref.astype(np.float64) - out)
            max_df = np.max(df)
            assert(max_df <= 1)


    def test_image_resize_channels(self, lenna_image, device_type, channel_format, channels):
        format = channel_format
        img_dtype = mp.kFloat32
        mode = mp.kBicubic

        origin, mp_origin = pre_process_image(lenna_image, device_type, channel_format, img_dtype, channels)

        sizes = [(240, 160), (320, 240), (480, 360), (1280, 720), (1920, 1080), (3840, 2160)]

        for size in sizes:
            ref = cv2.resize(origin, size, interpolation=cv2_resize_interp(mode))
            mp_out = mp.img.resize(mp_origin, size[0], size[1], mode=mode, format=format)
            if format == mp.kNCHW:
                mp_out = mp_out.permute((0, 2, 3, 1)) # convert to NHWC
            mp_out.squeeze_()
            out = mp_out.cpu().numpy()

            df = np.abs(ref.astype(np.float64) - out)
            max_df = np.max(df)
            assert(max_df <= 1)



class TestImageTransform(object):
    @pytest.fixture(params=[pytest.param(m, id=repr(m[0])) for m in 
            ((mp.kRotate0, -1),
             (mp.kRotate90, cv2.ROTATE_90_CLOCKWISE),
             (mp.kRotate180, cv2.ROTATE_180),
             (mp.kRotate270, cv2.ROTATE_90_COUNTERCLOCKWISE))])
    def rotate_mode(self, request):
        return request.param


    @pytest.fixture()
    def with_half(self):
        return True


    def test_rotate(self, lenna_image, rotate_mode, device_type, img_dtype, channel_format):
        lenna_image = cv2.resize(lenna_image, (1280, 720))
        origin, mp_origin = pre_process_image(lenna_image, device_type, channel_format, img_dtype)

        mode, code = rotate_mode
        if mode == mp.kRotate0:
            cv_out = origin
        else:
            cv_out = cv2.rotate(origin, code)

        mp_out = mp.img.rotate(mp_origin, mode, format=channel_format)
        
        if channel_format == mp.kNCHW:
            mp_out = mp_out.permute((0, 2, 3, 1)) # convert to NHWC
        mp_out = mp_out.cpu().numpy()

        df = np.abs(cv_out.astype(np.float64) - mp_out)
        max_df = np.max(df)
        assert(max_df == 0)


    @pytest.fixture(params=[pytest.param(m, id=repr(m)) for m in (mp.kVertical, mp.kHorizontal, mp.kHorizontalAndVertical)])
    def axes(self, request):
        return request.param


    def test_mirror(self, lenna_image, axes, device_type, img_dtype, channel_format):
        lenna_image = cv2.resize(lenna_image, (1280, 720))
        origin, mp_origin = pre_process_image(lenna_image, device_type, channel_format, img_dtype)

        if axes == mp.kVertical:
            cv_out = cv2.flip(origin, flipCode=0)
        elif axes == mp.kHorizontal:
            cv_out = cv2.flip(origin, flipCode=1)
        else:
            cv_out = cv2.flip(origin, flipCode=0)
            cv_out = cv2.flip(cv_out, flipCode=1)

        mp_out = mp.img.mirror(mp_origin, axes, format=channel_format)
        
        if channel_format == mp.kNCHW:
            mp_out = mp_out.permute((0, 2, 3, 1)) # convert to NHWC
        mp_out = mp_out.cpu().numpy()

        df = np.abs(cv_out.astype(np.float64) - mp_out)
        max_df = np.max(df)
        assert(max_df == 0)


    def test_normalize(self, lenna_image, device_type, img_dtype, channel_format):
        lenna_image = cv2.resize(lenna_image, (480, 320))
        origin, mp_origin = pre_process_image(lenna_image, device_type, channel_format, img_dtype)

        # (origin - mean) / std
        mean = np.array([110, 120, 130], dtype=np.float32).reshape((1, 1, 3))
        std = np.array([255/2, 255/2.1, 255/2.2], dtype=np.float32).reshape((1, 1, 3))
        std = np.array([1, 1, 1], dtype=np.float32).reshape((1, 1, 3))

        ref = origin.copy().astype(np.float32)
        ref -= mean
        ref /= std

        mp_mean = mp.from_numpy(mean).to(mp_origin.device).squeeze()
        mp_std = mp.from_numpy(std).to(mp_origin.device).squeeze()

        mp_out = mp.img.normalize(mp_origin, mp_mean, mp_std, format=channel_format)

        if channel_format == mp.kNCHW:
            mp_out = mp_out.permute((0, 2, 3, 1)) # convert to NHWC
        mp_out = mp_out.cpu().numpy()

        df = np.abs(ref.astype(np.float64) - mp_out)
        max_df = np.max(df)
        assert(max_df == 0)
