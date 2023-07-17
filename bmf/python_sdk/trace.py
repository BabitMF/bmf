import inspect
from enum import IntEnum
from bmf.lib._bmf import engine

# BMF Trace utilities

TraceType = engine.TraceType

TracePhase = engine.TracePhase


class TraceInfo:

    def __init__(self):
        self.data = ''

    def set_string(self, key, value):
        self.data += ',' + key + ':0:' + value

    def set_int(self, key, value):
        self.data += ',' + key + ':1:' + str(value)

    def set_float(self, key, value):
        self.data += ',' + key + ':2:' + str(value)


# Trace APIs


# Standard interface for trace
def BMF_TRACE(trace_type, name, trace_phase):
    caller = inspect.getouterframes(inspect.currentframe(), 2)
    engine.trace(int(trace_type), name, int(trace_phase), caller[1][3])


# Initialize trace (denotes the starting time for calculation)
def BMF_TRACE_INIT():
    caller = inspect.getouterframes(inspect.currentframe(), 2)
    engine.trace(int(TraceType.TRACE_START), 'Trace Start',
                 int(TracePhase.NONE), caller[1][3])


# For including user info
def BMF_TRACE_INFO(trace_type, name, trace_phase, trace_info):
    caller = inspect.getouterframes(inspect.currentframe(), 2)
    engine.trace_info(int(trace_type), name, int(trace_phase), trace_info.data,
                      caller[1][3])


# For formatting binary logs to tracelog (JSON log)
def BMF_TRACE_DONE():
    engine.trace_done()
