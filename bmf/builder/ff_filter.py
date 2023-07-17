from .bmf_stream import stream_operator
from .bmf_node import BmfNode
from .bmf_modules import bmf_modules


def get_filter_para(*args, **kwargs):
    out_args = [str(x) for x in args]
    out_kwargs = ['{}={}'.format(k, kwargs[k]) for k in sorted(kwargs)]

    params = out_args + out_kwargs

    return ":".join(params)


## @ingroup pyAPI
## @defgroup transFunc transcode functions
###@{
# BMF transcode related functions, can be called directly by BmfStream object
###@}


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  Build-in filter BMF stream
#  @param filter_name: the filter name in the libavfilter
#  @param args: the arguments for the filter
#  @param kwargs: the extra arguments for the filter stream such as: alias, stream_alias, type, path, entry
#  @return A BMF stream
@stream_operator()
def ff_filter(streams, filter_name, *args, **kwargs):
    ###@}
    alias = None
    stream_alias = None
    type = ""
    path = ""
    entry = ""
    if 'alias' in kwargs:
        alias = kwargs['alias']
        del kwargs['alias']
    if 'stream_alias' in kwargs:
        stream_alias = kwargs['stream_alias']
        del kwargs['stream_alias']
    if 'type' in kwargs:
        type = kwargs['type']
        del kwargs['type']
    if 'path' in kwargs:
        path = kwargs['path']
        del kwargs['path']
    if 'entry' in kwargs:
        entry = kwargs['entry']
        del kwargs['entry']

    para = get_filter_para(*args, **kwargs)
    if para is not None and len(para) > 0:
        option = {'name': filter_name, 'para': para}
    else:
        option = {'name': filter_name}

    if alias is not None:
        option['alias'] = alias

    module_info = {
        "name": bmf_modules['ff_filter'],
        "type": type,
        "path": path,
        "entry": entry
    }
    # create node
    return BmfNode(module_info, option, streams,
                   'immediate').stream(stream_alias=stream_alias)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'vflip' ffmpeg filter stream
@stream_operator()
def vflip(stream, **kwargs):
    ###@}
    return ff_filter(stream, 'vflip', **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'scale' ffmpeg filter stream
@stream_operator()
def scale(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'scale', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'setsar' ffmpeg filter stream
@stream_operator()
def setsar(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'setsar', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'pad' ffmpeg filter stream
@stream_operator()
def pad(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'pad', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'trim' ffmpeg filter stream
@stream_operator()
def trim(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'trim', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'setpts' ffmpeg filter stream
@stream_operator()
def setpts(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'setpts', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'loop' ffmpeg filter stream
@stream_operator()
def loop(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'loop', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'split' ffmpeg filter stream
@stream_operator()
def split(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'split', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  an 'adelay' ffmpeg filter stream
@stream_operator()
def adelay(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'adelay', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  an 'atrim' ffmpeg filter stream
@stream_operator()
def atrim(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'atrim', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  an 'amix' ffmpeg filter stream
@stream_operator()
def amix(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'amix', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  an 'afade' ffmpeg filter stream
@stream_operator()
def afade(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'afade', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  an 'asetpts' ffmpeg filter stream
@stream_operator()
def asetpts(stream, *args, **kwargs):
    ###@}
    return ff_filter(stream, 'asetpts', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  an 'overlay' ffmpeg filter stream
@stream_operator()
def overlay(stream, overlay_stream, *args, **kwargs):
    ###@}
    return ff_filter([stream, overlay_stream], 'overlay', *args, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'concat' ffmpeg filter stream
@stream_operator()
def concat(*streams, **kwargs):
    ###@}
    #'''
    #video_stream_count = kwargs.get('v', 1)
    #audio_stream_count = kwargs.get('a', 0)
    #stream_count = video_stream_count + audio_stream_count
    #if len(streams) % stream_count != 0:
    #    raise ValueError(
    #        'Expected concat input streams to have length multiple of {} (v={}, a={}); got {}'.format(
    #            stream_count, video_stream_count, audio_stream_count, len(streams)
    #        )
    #    )
    #seg_count = int(len(streams) / stream_count)

    #frame_seq_out = []

    #for video_stream_start in range(video_stream_count):
    #    video_streams = []
    #    video_stream_idx = video_stream_start
    #    while video_stream_idx < len(streams):
    #        video_streams.append(streams[video_stream_idx])
    #        video_stream_idx += stream_count
    #    frame_seq_out.append(module(video_streams, 'frame_sequencer', {}, 'immediate'))

    #for audio_stream_start in range(audio_stream_count):
    #    audio_streams = []
    #    audio_stream_idx = audio_stream_start + video_stream_count
    #    while audio_stream_idx < len(streams):
    #        audio_streams.append(streams[audio_stream_idx])
    #        audio_stream_idx += stream_count
    #    frame_seq_out.append(module(audio_streams, 'frame_sequencer', {}, 'immediate'))

    #seq_streams = []
    #for i in range(seg_count):
    #    for j in range(len(frame_seq_out)):
    #        seq_streams.append(frame_seq_out[j][i])

    #return ff_filter(seq_streams, 'concat', n=seg_count, v=video_stream_count, a=audio_stream_count)
    #'''
    return ff_filter(streams, 'concat', **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  @return  a 'fps' ffmpeg filter stream
@stream_operator()
def fps(stream, f, **kwargs):
    ###@}
    return ff_filter(stream, 'fps', fps=f, **kwargs)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  Build-in encoder BMF stream
#  Include av encoder and muxer
#  @param video_stream: the stream of video, it should be the first input stream of the encoder
#  @param audio_stream: the stream of audio
#  @param encoder_para: the parameters for the encoder
#  @return A BMF stream(s)
@stream_operator()
def encode(video_stream,
           audio_stream,
           encoder_para,
           type="",
           path="",
           entry="",
           stream_alias=None):
    ###@}
    module_info = {
        "name": bmf_modules['ff_encoder'],
        "type": type,
        "path": path,
        "entry": entry
    }
    if audio_stream is not None:
        return BmfNode(module_info,
                       encoder_para, [video_stream, audio_stream],
                       'immediate',
                       scheduler=1).stream(stream_alias=stream_alias)
    elif video_stream is not None:
        return BmfNode(module_info,
                       encoder_para, [video_stream],
                       'immediate',
                       scheduler=1).stream(stream_alias=stream_alias)
    else:  #both of the input streams are none, used in dynamical graph
        return BmfNode(bmf_modules['ff_encoder'],
                       encoder_para,
                       None,
                       'immediate',
                       scheduler=1).stream(stream_alias=stream_alias)


## @ingroup pyAPI
## @ingroup transFunc
###@{
#  A graph function to provide a build-in decoder BMF stream
#  Include av demuxer and decoder
#  @param decoder_para: the parameters for the decoder
#  @return A BMF stream(s)
@stream_operator()
def decode(self,
           decoder_para=None,
           type="",
           path="",
           entry="",
           stream_alias=None):
    ###@}
    module_info = {
        "name": bmf_modules['ff_decoder'],
        "type": type,
        "path": path,
        "entry": entry
    }
    if decoder_para is None:
        decoder_para = {}
    return BmfNode(module_info, decoder_para, self,
                   'immediate').stream(stream_alias=stream_alias)
