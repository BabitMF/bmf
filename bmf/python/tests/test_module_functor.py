
import bmf.hml.hmp as mp
import bmf

class MockDecoderModule(bmf.Module):
    def __init__(self, node=None, option=None):
        super().__init__(node=node, option=option)
        if 'mark_done' in option:
            self.mark_done = option['mark_done']
        else:
            self.mark_done = False

        if 'gen_eof' in option:
            self.gen_eof = option['gen_eof']
        else:
            self.gen_eof = False

        if 'npkts' in option:
            self.npkts = option['npkts']
        else:
            self.npkts = 1


    def process(self, task):
        if self.mark_done:
            task.timestamp = bmf.Timestamp.DONE
            return 0

        if self.gen_eof:
            task.fill_output_packet(0, bmf.Packet.generate_eof_packet())
            return 0

        for i in range(self.npkts):
            rgbformat = mp.PixelInfo(mp.kPF_RGB24)
            vf = bmf.VideoFrame(1920, 1080, rgbformat)
            task.fill_output_packet(0, bmf.Packet(vf))
        return 0


class MockProbeModule(bmf.Module):
    def __init__(self, node=None, option=None):
        super().__init__(node=node, option=option)

    def process(self, task):
        pkt = task.pop_packet_from_input_queue(0)
        vf = pkt.get(bmf.VideoFrame)
        info = {
            'width': vf.width,
            'height': vf.height,
        }

        task.fill_output_packet(0, bmf.Packet(info))
        return 0


def test_make_sync_func():
    decoder = bmf.make_sync_func(name='MockDecoderModule',
                       entry='test_module_functor.MockDecoderModule',
                       itypes=(), otypes=(bmf.VideoFrame,))
    probe = bmf.make_sync_func(name='MockProbeModule',
                       entry='test_module_functor.MockProbeModule',
                       itypes=(bmf.VideoFrame,), otypes=(dict,))
    assert(decoder is not None)
    assert(probe is not None)

    vf, = decoder()
    assert(isinstance(vf, bmf.VideoFrame))
    info, = probe(vf)
    assert(isinstance(info, dict))
    assert(info['width'] == 1920)
    assert(info['height'] == 1080)


def test_make_sync_func_exception():
    def expect_raise(f, e):
        try:
            f()
            assert(False)
        except e as v:
            return v

    decoder0 = bmf.make_sync_func(name='MockDecoderModule',
                       entry='test_module_functor.MockDecoderModule',
                       itypes=(), otypes=(bmf.VideoFrame,),
                       option={'gen_eof': False, 'mark_done': True})
    e = expect_raise(decoder0, bmf.ProcessDone)
    assert(str(e) == "Task done")


    decoder1 = bmf.make_sync_func(name='MockDecoderModule',
                       entry='test_module_functor.MockDecoderModule',
                       itypes=(), otypes=(bmf.VideoFrame,),
                       option={'gen_eof': True, 'mark_done': False})
    e = expect_raise(decoder1, bmf.ProcessDone)
    assert(str(e) == "Receive EOF packet")


def test_make_sync_func_irregular_outputs():
    decoder0 = bmf.make_sync_func(name='MockDecoderModule',
                       entry='test_module_functor.MockDecoderModule',
                       itypes=(), otypes=(bmf.VideoFrame,),
                       option={'npkts': 0})
    out0 = decoder0.execute().fetch(0)
    assert(len(out0) == 0)

    decoder1 = bmf.make_sync_func(name='MockDecoderModule',
                       entry='test_module_functor.MockDecoderModule',
                       itypes=(), otypes=(bmf.VideoFrame,),
                       option={'npkts': 2})
    out1 = decoder1.execute().fetch(0)
    assert(len(out1) == 2)
                    


if __name__ == '__main__':
    test_make_sync_func()
    test_make_sync_func_exception()
    test_make_sync_func_irregular_outputs()


