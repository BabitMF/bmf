import pytest
import numpy as np
import os
from hml_fixtures import has_cuda, has_ffmpeg
from hml_fixtures import mp, get_data_file
from test_color_convert import yuv_rgb, infer_pix_info

videos = [
        ('H420', (720, 1280), get_data_file("videos/H420.mp4")),
        ('H422', (720, 1280), get_data_file("videos/H422.mp4")),
        ('H444', (720, 1280), get_data_file("videos/H444.mp4")),
    ]

@pytest.fixture(params=[pytest.param(cfg, id=repr(cfg[0])) for cfg in videos])
def video_config(request):
    return request.param


@pytest.mark.skipif(not has_ffmpeg, reason="need enable ffmpeg")
class TestFFMPEGInterOp(object):
    def test_ffmpeg_interop(self, video_config):
        fmt, shape, fn = video_config
        pix_info = infer_pix_info(fmt)

        reader = mp.ffmpeg.VideoReader(fn)
        frames = reader.read(16)

        width, height = frames[0].width(), frames[0].height()

        assert(frames[0].format() == pix_info.format)
        assert(len(frames) == 16)
        assert((width, height) == shape)

        writer = mp.ffmpeg.VideoWriter("tmp.mp4", width, height, 30, pix_info)
        writer.write(frames)


class TestFrame(object):
    @pytest.fixture
    def frame_info(self, yuv_rgb):
        fmt, yuv, _, rgb = yuv_rgb
        mp_yuv = [mp.from_numpy(p) for p in yuv]
        frame = mp.Frame(mp_yuv, infer_pix_info(fmt))
        return fmt, frame, yuv, rgb


    @pytest.mark.skipif(not has_cuda, reason="need enable cuda")
    def test_frame_make(self,):
        h420 = infer_pix_info('H420')
        nv21 = infer_pix_info('NV21')

        frame0 = mp.Frame(1920, 1080, h420, device="cuda:0")
        frame1 = mp.Frame(1920, 1080, nv21, device="cuda:0")

        assert(frame0.width() == 1920)
        assert(frame0.height() == 1080)
        assert(frame0.nplanes() == 3)
        assert(frame0.plane(1).size(0) == 1080//2)
        assert(frame0.plane(1).size(1) == 1920//2)
        assert(frame0.plane(1).size(2) == 1)
        assert(frame0.plane(2).size(0) == 1080//2)
        assert(frame0.plane(2).size(1) == 1920//2)
        assert(frame0.plane(2).size(2) == 1)
        assert(frame0.format() == mp.kPF_YUV420P)
        assert(frame0.dtype() == mp.uint8)
        assert(frame0.device().type() == mp.kCUDA)

        assert(frame1.width() == 1920)
        assert(frame1.height() == 1080)
        assert(frame1.nplanes() == 2)
        assert(frame1.plane(1).size(0) == 1080//2)
        assert(frame1.plane(1).size(1) == 1920//2)
        assert(frame1.plane(1).size(2) == 2)
        assert(frame1.format() == mp.kPF_NV21)
        assert(frame1.dtype() == mp.uint8)
        assert(frame1.device().type() == mp.kCUDA)


    def test_frame(self, frame_info):
        fmt, frame, yuv, rgb = frame_info
        pix_info = infer_pix_info(fmt)
        mp_yuv = [mp.from_numpy(p) for p in yuv]

        width, height = yuv[0].shape[1], yuv[0].shape[0]
        frame = mp.Frame(mp_yuv, width, height, pix_info)
        assert(frame.width() == width)
        assert(frame.height() == height)
        assert(frame.nplanes() == len(yuv))
        assert(frame.format() == pix_info.format)
        assert(frame.dtype() == mp_yuv[0].dtype)
        assert(frame.device().type() == mp.kCPU)

        for i in range(frame.nplanes()):
            d = frame.plane(i).squeeze().numpy().reshape(yuv[i].shape)
            assert(np.allclose(d, yuv[i]))

        # data convert
        if has_cuda:
            tmp = frame.to(mp.kCUDA).to(mp.kCPU)
            assert(tmp.pix_info().format == pix_info.format)
            assert(tmp.pix_info().space == pix_info.space)
            for i in range(tmp.nplanes()):
                d = tmp.plane(i).squeeze().numpy().reshape(yuv[i].shape)
                assert(np.allclose(d, yuv[i]))

        # crop
        tmp = frame.crop(100, 200, 150, 300)
        assert(tmp.pix_info().format == pix_info.format)
        assert(tmp.pix_info().space == pix_info.space)
        for i in range(frame.nplanes()):
            p = frame.plane(i)
            w, h = p.shape[1], p.shape[0]
            wscale = frame.width() // w
            hscale = frame.height() // h
            left = 100//wscale
            width = 150//wscale
            top = 200//hscale
            height = 300//hscale
            ref = yuv[i][top:top+height, left:left+width]
            d = tmp.plane(i).squeeze().numpy().reshape(ref.shape)
            assert(np.allclose(d, ref))

        # image interop
        image = frame.to_image(mp.kNHWC)
        assert(image.width() == frame.width())
        assert(image.height() == frame.height())
        assert(image.nchannels() == 3)
        diff = np.abs(image.data().view(rgb.shape).numpy() - rgb.astype(np.float32))
        assert(np.max(diff) < 20) # only ensure almost equal

        tmp = mp.Frame.from_image(image, pix_info)
        for i in range(tmp.nplanes()):
            d = tmp.plane(i).squeeze().numpy().reshape(yuv[i].shape)
            diff = np.abs(d.astype(np.float32) - yuv[i])
            assert(np.max(diff) < 30) # only ensure almost equal


    def test_frame_seq(self, yuv_rgb):
        fmt, yuv, _, rgb = yuv_rgb
        pix_info = infer_pix_info(fmt)
        batch = 4
        mp_yuv = [mp.from_numpy(p) for p in yuv]
        yuv_batch = [np.vstack([p[np.newaxis, ...] + i for i in range(batch)]) for p in yuv] 

        frames = []
        for i in range(batch):
            tmp = [mp.from_numpy(p[i]) for p in yuv_batch]
            frames.append(mp.Frame(tmp, pix_info))
        frame_seq = mp.concat(frames)

        assert(frame_seq.width() == yuv[0].shape[1])
        assert(frame_seq.height() == yuv[0].shape[0])
        assert(frame_seq.nplanes() == len(yuv))
        assert(frame_seq.format() == pix_info.format)
        assert(frame_seq.dtype() == mp_yuv[0].dtype)
        assert(frame_seq.device().type() == mp.kCPU)

        for i in range(frame_seq.nplanes()):
            ref = yuv_batch[i]
            d = frame_seq.plane(i).squeeze().numpy().reshape(ref.shape)
            assert(np.allclose(d, ref))

        # getitem
        frame = frame_seq[0]
        for i in range(frame.nplanes()):
            ref = yuv[i]
            d = frame.plane(i).squeeze().numpy().reshape(ref.shape)
            assert(np.allclose(d, ref))

        # data convert
        if has_cuda:
            tmp = frame_seq.to(mp.kCUDA).to(mp.kCPU)
            for i in range(tmp.nplanes()):
                ref = yuv_batch[i]
                d = frame_seq.plane(i).squeeze().numpy().reshape(ref.shape)
                assert(np.allclose(d, ref))

        # crop
        tmp = frame_seq.crop(100, 200, 150, 300)
        for i in range(frame_seq.nplanes()):
            p = frame_seq.plane(i)
            w, h = p.shape[2], p.shape[1]
            wscale = frame.width() // w
            hscale = frame.height() // h
            left = 100//wscale
            width = 150//wscale
            top = 200//hscale
            height = 300//hscale
            ref = yuv_batch[i][:, top:top+height, left:left+width]
            d = tmp.plane(i).squeeze().numpy().reshape(ref.shape)
            assert(np.allclose(d, ref))



class TestImage(object):
    @pytest.mark.skipif(not has_cuda, reason="need enable cuda")
    def test_image_make(self,):
        image0 = mp.Image(1920, 1080, 3, mp.kNCHW, device="cuda:0", dtype=mp.uint16)
        image1 = mp.Image(1920, 1080, 3, mp.kNHWC, device="cuda:0", dtype=mp.uint16)

        assert(image0.width() == 1920)
        assert(image0.height() == 1080)
        assert(image0.nchannels() == 3)
        assert(image0.format() == mp.kNCHW)
        assert(image0.dtype() == mp.uint16)
        assert(image0.device().type() == mp.kCUDA)

        assert(image1.width() == 1920)
        assert(image1.height() == 1080)
        assert(image1.nchannels() == 3)
        assert(image1.format() == mp.kNHWC)
        assert(image1.dtype() == mp.uint16)
        assert(image1.device().type() == mp.kCUDA)


    def test_image(self, yuv_rgb):
        fmt, yuv, _, rgb = yuv_rgb
        mp_rgb = mp.from_numpy(rgb)

        image = mp.Image(mp_rgb, mp.kNHWC)
        assert(image.format() == mp.kNHWC)
        assert(image.wdim() == 1)
        assert(image.hdim() == 0)
        assert(image.cdim() == 2)
        assert(image.width() == rgb.shape[1])
        assert(image.height() == rgb.shape[0])
        assert(image.nchannels() == rgb.shape[2])
        assert(image.device().type() == mp.kCPU)

        #
        if has_cuda:
            tmp = image.to(mp.kCUDA).to(mp.kCPU)
            assert(np.allclose(rgb, image.data().numpy()))

        #
        tmp = image.to(mp.kNCHW)
        assert(tmp.wdim() == 2)
        assert(tmp.hdim() == 1)
        assert(tmp.cdim() == 0)

        #
        tmp = image.crop(100, 150, 200, 300).data()
        assert(np.allclose(tmp.numpy(), rgb[150:450, 100:300]))

        for i in range(image.nchannels()):
            ref = rgb[:, :, i]
            tmp = image.select(i).data().view(ref.shape).numpy()
            assert(np.allclose(tmp, ref))


    def test_image_seq(self, yuv_rgb):
        fmt, yuv, _ ,rgb = yuv_rgb

        batch = 4
        rgbs = np.vstack([rgb[np.newaxis, ...] + i for i in range(batch)])
        images = [mp.Image(mp.from_numpy(rgbs[i]), mp.kNHWC) for i in range(batch)]
        image_seq = mp.concat(images)

        assert(image_seq.format() == mp.kNHWC)
        assert(image_seq.wdim() == 2)
        assert(image_seq.hdim() == 1)
        assert(image_seq.cdim() == 3)
        assert(image_seq.width() == rgb.shape[1])
        assert(image_seq.height() == rgb.shape[0])
        assert(image_seq.nchannels() == rgb.shape[2])
        assert(image_seq.device().type() == mp.kCPU)

        #
        if has_cuda:
            tmp = image_seq.to(mp.kCUDA).to(mp.kCPU)
            assert(np.allclose(rgbs, image_seq.data().numpy()))

        #
        tmp = image_seq.to(mp.kNCHW)
        assert(tmp.wdim() == 3)
        assert(tmp.hdim() == 2)
        assert(tmp.cdim() == 1)

        #
        tmp = image_seq.crop(100, 150, 200, 300).data()
        assert(np.allclose(tmp.numpy(), rgbs[:, 150:450, 100:300]))

        for i in range(image_seq.nchannels()):
            ref = rgbs[:, :, :, i]
            tmp = image_seq.select(i).data().view(ref.shape).numpy()
            assert(np.allclose(tmp, ref))

        tmp = image_seq.slice(1, 3)
        assert(np.allclose(tmp.data().numpy(), rgbs[1:3]))

        #
        tmp = image_seq[0]
        assert(np.allclose(tmp.data().numpy(), rgbs[0]))