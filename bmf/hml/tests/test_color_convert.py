
import pytest
import numpy as np
import os
from hml_fixtures import device_type
from hml_fixtures import mp, TEST_DATA_ROOT
import glob


#### 
def read_rgb(fn, width, height):
    rgb = np.fromfile(fn, dtype=np.uint8).reshape((height, width, 3))
    return rgb


def read_yuv(fn, width, height, fmt):
    yuv = np.fromfile(fn, dtype=np.uint8)
    if fmt[-3:] == "420":
        i = 0
        y = yuv[i:i+width*height].reshape((height, width, 1))
        i += width*height
        u = yuv[i:i+width*height//4].reshape((height//2, width//2, 1))
        i += width*height//4
        v = yuv[i:i+width*height//4].reshape((height//2, width//2, 1))
        yuv = [y, u, v]
    elif fmt[-3:] == "422":
        i = 0
        y = yuv[i:i+width*height].reshape((height, width, 1))
        i += width*height
        u = yuv[i:i+width*height//2].reshape((height, width//2, 1))
        i += width*height//2
        v = yuv[i:i+width*height//2].reshape((height, width//2, 1))
        yuv = [y, u, v]
    elif fmt[-3:] == "444":
        yuv = yuv.reshape((3, width, height, 1))
        yuv = [yuv[0], yuv[1], yuv[2]]
    elif fmt in ["NV12", "NV21"]:
        i = 0
        y = yuv[:width*height].reshape((height, width, 1))
        i += width*height
        uv = yuv[i:i+width*height//2].reshape((height//2, width//2, 2))
        yuv =[y, uv]
    else:
        assert(False)
    return yuv

    
def read_yuv_rgb(width, height, yuv_fn0, yuv_fn1, rgb_fn):
    fmt = os.path.basename(yuv_fn0)[:-4]
    yuv0 = read_yuv(yuv_fn0, width, height, fmt)
    yuv1 = read_yuv(yuv_fn1, width, height, fmt)
    rgb = read_rgb(rgb_fn, width, height)
    return fmt, yuv0, yuv1, rgb


def get_yuv_rgb_pairs():
    fns = glob.glob(TEST_DATA_ROOT + "/colors/*.yuv")
    pairs = []
    for fn in fns:
        n = os.path.basename(fn)
        if "RGB24" not in n and "YUV" not in n:
            rgb_fn = fn[:-4] + "_RGB24.yuv"
            fn1 = fn[:-4] + "_YUV.yuv"
            format, yuv0, yuv1, rgb = read_yuv_rgb(800, 800, fn, fn1, rgb_fn)
            yield format, yuv0, yuv1, rgb


@pytest.fixture(params=[pytest.param(p, id=str(p[0]))
                for p in get_yuv_rgb_pairs()],
                scope='session')
def yuv_rgb(request):
    return request.param


def max_diff(a, b):
    df = np.abs(a.astype(np.float32) - b)
    return np.mean(df), np.max(df)


def infer_pix_info(fmt):
    cs = mp.kCS_BT709 if fmt[-4] == 'H' else mp.kCS_BT470BG

    if fmt[-3:] == '420':
        return mp.PixelInfo(mp.kPF_YUV420P, cs,  mp.kCR_MPEG)
    elif fmt[-3:] == '422':
        return mp.PixelInfo(mp.kPF_YUV422P, cs,  mp.kCR_MPEG)
    elif fmt[-3:] == '444':
        return mp.PixelInfo(mp.kPF_YUV444P, cs,  mp.kCR_MPEG)
    elif fmt == 'NV12':
        return mp.PixelInfo(mp.kPF_NV12, cs,  mp.kCR_MPEG)
    elif fmt == 'NV21':
        return mp.PixelInfo(mp.kPF_NV21, cs,  mp.kCR_MPEG)
    else:
        assert(False)


# NOTE: only test uint8
def test_yuv_rgb_convert(yuv_rgb, device_type):
    format, yuv0, yuv1, rgb = yuv_rgb

    im_yuv0 = [mp.from_numpy(c).to(device_type) for c in yuv0]
    im_yuv1 = [mp.from_numpy(c).to(device_type) for c in yuv1]
    im_rgb = mp.from_numpy(rgb).to(device_type)

    v_rgb = mp.img.yuv_to_rgb(im_yuv0, pix_info=infer_pix_info(format), cformat=mp.kNHWC)
    v_yuv = mp.img.rgb_to_yuv(im_rgb, pix_info=infer_pix_info(format), cformat=mp.kNHWC)
    
    mean_df, max_df = max_diff(v_rgb.cpu().numpy(), rgb)
    assert(mean_df < 1)
    if format in ["NV12", "NV21"]:
        assert(max_df <= 14) #FIXME
    else:
        assert(max_df <= 2)

    #Note: 为什么要用yuv1？因为yuv2rgb由于有clip操作，所以变换不可逆
    #      使用yuv1(rgb -> yuv1), 使得对比具有一致性
    assert(len(v_yuv) == len(yuv1))
    y_mean_df, y_max_df = max_diff(v_yuv[0].cpu().numpy(), yuv1[0])
    assert(y_max_df <= 1)

    for i in range(1, len(yuv1)):
        mean_df, max_df = max_diff(v_yuv[i].cpu().numpy(), yuv1[i])
        assert(max_df <= 10) #FIXME
        assert(mean_df <= 1)


