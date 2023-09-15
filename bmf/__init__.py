import sys
import ctypes
import platform
from bmf.python_sdk.module_functor import make_sync_func
if platform.system().lower() != 'windows':
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

# import hmp types
import bmf.hml.hmp as mp
# import bmf_sdk types

from bmf.lib._bmf.sdk import VideoFrame, AudioFrame, Packet, BMFAVPacket
from bmf.lib._bmf.sdk import Task

#
from .ffmpeg_engine import FFmpegEngine
from .python_sdk import Module, LogLevel, Log, InputType, ProcessResult,  Timestamp, scale_av_pts, av_time_base, \
    SubGraph, BMF_TRACE, BMF_TRACE_INFO, BMF_TRACE_INIT, BMF_TRACE_DONE, TraceType, TracePhase, TraceInfo, get_version, get_commit, change_dmp_path
from .python_sdk import make_sync_func, ProcessDone
from .builder import BmfGraph, graph, BmfCallBackType, module, py_module, c_module, go_module, vflip, scale, setsar, pad, trim, setpts, loop, split, \
    adelay, atrim, amix, afade, asetpts, overlay, concat, fps, encode, decode, create_module, GraphMode, bmf_sync, \
    GraphConfig, get_module_file_dependencies, ff_filter
from .server import ServerGateway, ServerGatewayNew

if platform.system().lower() != 'windows':
    sys.setdlopenflags(flags)
