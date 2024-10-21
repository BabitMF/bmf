import pytest
import bmf.hmp as mp
import numpy as np
from bmf_fixtures import has_cuda, has_torch
from bmf.lib._bmf.sdk import VideoFrame, AudioFrame, BMFAVPacket
from bmf.lib._bmf.sdk import Packet
from bmf import MediaDesc, MediaType, bmf_convert
from bmf.lib._bmf import sdk


class TestVideoFrame(object):

    def test_image(self):
        # construct Image by parameters
        rgbformat = mp.PixelInfo(mp.kPF_RGB24)
        vf = VideoFrame(1920, 1080, rgbformat)

        assert (vf.defined())
        assert (vf.width == 1920)
        assert (vf.height == 1080)
        assert (vf.dtype == mp.uint8)
        assert (vf.device == mp.Device("cpu"))

        vf_sub = vf.crop(100, 200, 300, 400)
        assert (vf_sub.width == 300)
        assert (vf_sub.height == 400)

    def test_frame(self):
        # construct Frame by parameters
        H420 = mp.PixelInfo(mp.kPF_YUV420P, mp.kCS_BT709)
        vf = VideoFrame(1920, 1080, pix_info=H420)
        assert (vf.defined())
        assert (vf.width == 1920)
        assert (vf.height == 1080)
        _ = vf.frame()  # no exception

    def test_numpy_interop(self):
        # construct image from numpy array
        data_np = np.random.randint(255, size=(1080, 1920, 3), dtype=np.uint8)
        rgbformat = mp.PixelInfo(mp.kPF_RGB24)
        image = mp.Frame(mp.from_numpy(data_np), rgbformat)
        vf = VideoFrame(image)
        data_np_1 = vf.frame().plane(0).numpy()
        assert (np.allclose(data_np, data_np_1))

        # construct frame from numpy arrays
        H420 = mp.PixelInfo(mp.kPF_YUV420P, mp.kCS_BT709)
        Y = np.random.randint(255, size=(1080, 1920), dtype=np.uint8)
        U = np.random.randint(255, size=(540, 960), dtype=np.uint8)
        V = np.random.randint(255, size=(540, 960), dtype=np.uint8)
        yuv = [mp.from_numpy(v) for v in [Y, U, V]]
        frame = mp.Frame(yuv, pix_info=H420)
        vf = VideoFrame(frame)
        Y_1 = vf.frame().plane(0).squeeze_().numpy()
        U_1 = vf.frame().plane(1).squeeze_().numpy()
        V_1 = vf.frame().plane(2).squeeze_().numpy()
        assert (np.allclose(Y, Y_1))
        assert (np.allclose(U, U_1))
        assert (np.allclose(V, V_1))

    def test_opaque_data(self):
        rgbformat = mp.PixelInfo(mp.kPF_RGB24)
        vf = VideoFrame(1920, 1080, rgbformat)
        assert (vf.private_get(dict) is None)  # bmf_sdk::JsonParam

        ref = {'int': 42, 'str': 'hello', 'dict': {'a': 10, 'b': 11}}
        vf.private_attach(ref)

        data = vf.private_get(dict)
        assert (data is not None)
        assert (data['int'] == ref['int'])
        assert (data['str'] == ref['str'])
        assert (data['dict']['a'] == ref['dict']['a'])
        assert (data['dict']['b'] == ref['dict']['b'])

        assert (id(data)
                != id(ref))  # as we convert dict to JsonParam in c++ side

    @pytest.mark.skipif(not has_cuda, reason="CUDA is not enabled")
    def test_cuda_constructors(self):
        H420 = mp.PixelInfo(mp.kPF_YUV420P, mp.kCS_BT709)
        vf_frame = VideoFrame(1920, 1080, pix_info=H420, device='cuda:0')
        assert (vf_frame.device == mp.Device("cuda:0"))

    @pytest.mark.skipif(not has_cuda, reason="CUDA is not enabled")
    def test_cuda_interop(self):
        rgbformat = mp.PixelInfo(mp.kPF_RGB24)
        vf = VideoFrame(1920, 1080, rgbformat, device='cuda:0')

        stream = mp.create_stream(mp.kCUDA)
        with stream:  # set currrent stream
            data = vf.frame().plane(0)
            data_resized = mp.img.resize(data,
                                         4000,
                                         2000,
                                         mode=mp.kBicubic,
                                         format=mp.kNHWC)
            vf_out = VideoFrame(mp.Frame(data_resized, rgbformat))
            vf_out.record(
                use_current=True)  # record on current stream(default)
            assert (vf_out.ready() == False)
            assert (vf_out.stream == stream.handle())

        vf_out.synchronize()
        assert (vf_out.ready())


class TestAudioFrame(object):

    def test_constructors(self):
        af = AudioFrame(160, sdk.kLAYOUT_STEREO, True, dtype=mp.uint8)
        assert (af.defined())
        assert (af.layout == sdk.kLAYOUT_STEREO)
        assert (af.dtype == mp.uint8)
        assert (af.planer == True)
        assert (af.nsamples == 160)
        assert (af.nchannels == 2)
        assert (af.nplanes == 2)

        af.sample_rate = 44100
        assert (af.sample_rate == 44100)


class testBMFAVPacket(object):

    def test_constructors(self):
        av_pkt = BMFAVPacket(1024, dtype=mp.uint8)
        assert (av_pkt.defined())
        assert (av_pkt.nbytes == 1024)
        assert (av_pkt.data.dim == 1)
        assert (av_pkt.data.size(0) == 1024)


class TestPacket(object):

    def test_control_info(self):
        eos_pkt = Packet.generate_eos_packet()
        eof_pkt = Packet.generate_eof_packet()
        assert (eos_pkt.timestamp == sdk.kEOS)
        assert (eof_pkt.timestamp == sdk.kBMF_EOF)

        pkt = Packet(0)
        pkt.timestamp = 42  # set timestamp
        assert (pkt.timestamp == 42)

    def test_cpp_types(self):
        rgbformat = mp.PixelInfo(mp.kPF_RGB24)
        vf = VideoFrame(1920, 1080, rgbformat)
        pkt = Packet(vf)  #
        assert (pkt.is_(VideoFrame))  # type check
        v = pkt.get(VideoFrame)  # cast from pkt to VideoFrame
        assert (isinstance(v, VideoFrame))
        assert (v.width == 1920)
        assert (v.height == 1080)

        af = AudioFrame(160, sdk.kLAYOUT_STEREO, True, dtype=mp.uint8)
        pkt = Packet(af)
        assert (pkt.is_(AudioFrame))

        av_pkt = BMFAVPacket(1024, dtype=mp.uint8)
        pkt = Packet(av_pkt)
        assert (pkt.is_(BMFAVPacket))

    def test_py_types(self):
        # custom python object
        class Foo(object):
            pass

        foo = Foo()
        pkt = Packet(foo)
        assert (pkt.is_(None))  # type info is no needed
        assert (pkt.is_(Foo))  # or specify the exact type
        assert (not pkt.is_(str))  #
        v = pkt.get(None)  #
        assert (isinstance(v, Foo))

        # dict -> JsonParam
        maps = {"a": 10, "b": {"b.b": "hello"}}
        pkt = Packet(maps)
        assert (pkt.class_name == 'bmf_sdk::JsonParam')
        assert (not pkt.is_(None))
        assert (pkt.is_(dict))  # or specify the exact type
        v = pkt.get(dict)  #
        #v = pkt.get(str) # exception
        assert (v['a'] == 10)
        assert (v['b']['b.b'] == 'hello')

        # numpy
        arr = np.random.normal(size=1000)
        pkt = Packet(arr)
        assert (pkt.is_(np.ndarray))
        v = pkt.get(np.ndarray)
        assert (np.allclose(arr, v))

        # TODO: multi-thread test


class TestMediaDesc(object):

    def test_mediadesc(self):
        md = MediaDesc()
        md.width(1080).height(720).media_type(MediaType.kCVMat)
        md.pixel_format(mp.kPF_YUV420P).color_space(mp.kCS_BT709).device(
            mp.Device(mp.kCPU))
        assert (md.width() == 1080)
        assert (md.height() == 720)
        assert (md.media_type() == MediaType.kCVMat)
        assert (md.pixel_format() == mp.kPF_YUV420P)
        assert (md.color_space() == mp.kCS_BT709)
        assert (md.device().type() == mp.kCPU)

    def test_backend_convert(self):
        md = MediaDesc()
        md.width(1920).height(1080).pixel_format(mp.kPF_RGB24)

        H420 = mp.PixelInfo(mp.kPF_YUV420P, mp.kCS_BT709)
        vf = VideoFrame(640, 360, pix_info=H420)

        dst_vf = bmf_convert(vf, MediaDesc(), md)
        assert (dst_vf.width == 1920)
        assert (dst_vf.height == 1080)
        assert (dst_vf.frame().format() == mp.kPF_RGB24)

    @pytest.mark.skipif(not has_cuda, reason="CUDA is not enabled")
    def test_backend_convert_cuda(self):
        H420 = mp.PixelInfo(mp.kPF_YUV420P, mp.kCS_BT709)
        vf = VideoFrame(640, 360, pix_info=H420)

        md = MediaDesc()
        md.width(1920).height(1080).pixel_format(mp.kPF_RGB24).device(
            mp.Device("cuda:0"))

        dst_vf = bmf_convert(vf, MediaDesc(), md)
        assert (dst_vf.width == 1920)
        assert (dst_vf.height == 1080)
        assert (dst_vf.frame().format() == mp.kPF_RGB24)
        assert (dst_vf.frame().device().type() == mp.kCUDA)
        assert (dst_vf.frame().device().index() == 0)

    def test_backend_convert_numpy(self):
        H420 = mp.PixelInfo(mp.kPF_YUV420P, mp.kCS_BT709)
        vf = VideoFrame(640, 360, pix_info=H420)

        md = MediaDesc()
        md.width(1920).height(1080).pixel_format(mp.kPF_RGB24).media_type(
            MediaType.kTensor)

        dst_vf = bmf_convert(vf, MediaDesc(), md)
        assert (dst_vf.width == 1920)
        assert (dst_vf.height == 1080)
        assert (dst_vf.frame().format() == mp.kPF_RGB24)
        assert (bool(dst_vf) == True)

        np_array = dst_vf.private_get(np.ndarray)
        print(np_array.shape)
        assert (np_array.shape[0] == 1080)
        assert (np_array.shape[1] == 1920)
        assert (np_array.shape[2] == 3)

        src_vf = VideoFrame()
        src_vf.private_attach(np_array)

        src_md = MediaDesc()
        src_md.pixel_format(mp.kPF_RGB24).media_type(MediaType.kTensor)

        dst_md = MediaDesc()
        dst_md.pixel_format(mp.kPF_YUV420P).color_space(
            mp.kCS_BT709).width(320).height(180)

        new_vf = bmf_convert(src_vf, src_md, dst_md)

        assert new_vf.width == 320
        assert new_vf.height == 180
        assert new_vf.frame().format() == mp.kPF_YUV420P

    @pytest.mark.skipif(has_torch == 0, reason="torch is not enabled")
    def test_backend_convert_torch(self):
        H420 = mp.PixelInfo(mp.kPF_YUV420P, mp.kCS_BT709)
        vf = VideoFrame(640, 360, pix_info=H420)
        md = MediaDesc()
        md.width(1920).height(1080).pixel_format(mp.kPF_RGB24).media_type(
            MediaType.kATTensor)
        print("has_torch", has_torch)
        dst_vf = bmf_convert(vf, MediaDesc(), md)
        torch_tensor = dst_vf.private_get(torch.Tensor)
        assert (torch_tensor.shape[0] == 1080)
        assert (torch_tensor.shape[1] == 1920)
        assert (torch_tensor.shape[2] == 3)
        src_vf = VideoFrame()
        src_vf.private_attach(torch_tensor)

        src_md = MediaDesc()
        src_md.pixel_format(mp.kPF_RGB24).media_type(MediaType.kATTensor)

        dst_md = MediaDesc()
        dst_md.pixel_format(mp.kPF_YUV420P).color_space(
            mp.kCS_BT709).width(1280).height(720)

        new_vf = bmf_convert(src_vf, src_md, dst_md)

        assert (new_vf.width == 1280)
        assert (new_vf.height == 720)
        assert (new_vf.frame().format() == mp.kPF_YUV420P)


if __name__ == '__main__':
    pass
