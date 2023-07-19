import sys
from fractions import Fraction
import numpy
from bmf import Module, Log, LogLevel, InputType, ProcessResult, Packet, Timestamp, scale_av_pts, av_time_base, \
    BmfCallBackType, VideoFrame, AudioFrame
import libbmf_sdk as csdk

if sys.version_info.major == 2:
    TimeStampType = long
else:
    TimeStampType = int


class c_module(Module):

    def __init__(self, node, option=None):
        self.node_ = node

        if option is None:
            Log.log_node(LogLevel.ERROR, self.node_, "Option is none")
            return

        # get .so path
        if 'module_path' not in option.keys():
            Log.log_node(LogLevel.ERROR, self.node_, "No module path")
            return
        self.module_path_ = option['module_path']

        # get .so name and class name
        if 'module_entry' not in option.keys():
            Log.log_node(LogLevel.ERROR, self.node_, "No module entry")
            return
        self.module_entry_ = option['module_entry']
        ss = self.module_entry_.split(':')
        self.module_name_ = ss[0]
        self.class_name = ss[1]

        Log.log_node(LogLevel.ERROR, self.node_, "c module path:",
                     self.module_path_, ", module:", self.module_name_,
                     ", class:", self.class_name)

        # import module
        sys.path.append(self.module_path_)
        py_mod = __import__(self.module_name_)
        c_class = getattr(py_mod, self.class_name, None)

        # create c module option
        self.c_option_ = csdk.Option()
        for (k, v) in option.items():
            self.c_option_.set(k, v)

        # create c module
        self.c_mod_ = c_class(self.c_option_)
        Log.log_node(LogLevel.ERROR, self.node_, "c module:", self.c_mod_)

    # convert avframe(ffmpeg frame) to bmf frame
    def python_vframe_to_c_vframe(self, py_frame):
        planes = []

        # buffer pointer
        buffer_pointer = py_frame.planes[0].buffer_ptr

        prev_plane_size = 0
        for i, py_plane in enumerate(py_frame.planes):
            # copy video plane data from av frame to bmf frame
            planes.append(
                csdk.VideoPlane(py_plane.line_size, py_plane.width,
                                py_plane.height, py_plane.buffer_size,
                                py_plane.buffer_ptr))

        # create bmf frame
        c_frame = csdk.VideoFrame(py_frame.width, py_frame.height,
                                  py_frame.format.name, 0, buffer_pointer,
                                  planes)

        # set pts
        c_frame.set_pts(py_frame.pts)

        # set time base
        c_time_base = csdk.Rational(py_frame.time_base.numerator,
                                    py_frame.time_base.denominator)
        c_frame.set_time_base(c_time_base)

        return c_frame

    def python_aframe_to_c_aframe(self, py_frame):
        planes = []

        # change buffer pointer to planes[0].buffer_ptr
        # in pyav, the buffer_point is NULL
        buffer_pointer = py_frame.planes[0].buffer_ptr

        prev_plane_size = 0
        for i, py_plane in enumerate(py_frame.planes):
            # copy video plane data from av frame to bmf frame
            planes.append(
                csdk.AudioPlane(py_plane.buffer_size, py_plane.buffer_ptr))

        # create bmf frame
        c_frame = csdk.AudioFrame(py_frame.format.name, py_frame.layout.name,
                                  py_frame.samples, 0, buffer_pointer, planes)

        # set pts
        c_frame.set_pts(py_frame.pts)

        # set time base
        c_time_base = csdk.Rational(py_frame.time_base.numerator,
                                    py_frame.time_base.denominator)
        c_frame.set_time_base(c_time_base)
        c_frame.set_sample_rate(py_frame.sample_rate)
        return c_frame

    def trans2dict(self, data):
        if "__dict__" in dir(data):
            result = {}
            for key, value in data.__dict__.items():
                result[key] = self.trans2dict(value)
        else:
            if isinstance(data, int) or \
                    isinstance(data, str) or \
                    isinstance(data, float) or \
                    isinstance(data, bool):
                result = data
            else:
                result = None
                raise (RuntimeError("not support type:" + str(type(data))))
        return result

    def python_packet_to_c_packet(self, py_pkt):
        c_pkt = csdk.Packet()
        c_pkt.set_timestamp(TimeStampType(py_pkt.get_timestamp()))

        # for EOF packet, ignore data
        if py_pkt.get_timestamp() == Timestamp.EOF:
            return c_pkt

        py_data = py_pkt.get_data()

        # TODO: support other type as dict and so on
        if isinstance(py_data, VideoFrame):
            c_frame = self.python_vframe_to_c_vframe(py_data)
            c_pkt.py_set_data(c_frame)
            return c_pkt
        elif isinstance(py_data, AudioFrame):
            c_frame = self.python_aframe_to_c_aframe(py_data)
            c_pkt.py_set_data(c_frame)
            return c_pkt
        elif isinstance(py_data, numpy.ndarray):
            c_pkt.py_set_data(py_data)
            return c_pkt
        else:
            try:
                c_frame = self.trans2dict(py_data)
            except RuntimeError as e:
                Log.log_node(LogLevel.ERROR, self.node_, e.args)
                return None
            c_pkt.py_set_data(c_frame)
            return c_pkt

    def c_vframe_to_python_vframe(self, c_frame):
        buf_pointers = []
        line_size = []
        last_plane_size = 0

        # prepare buffer pointer and line size
        for c_plane in c_frame.py_get_planes():
            buf_pointers.append(c_frame.py_get_buffer() + last_plane_size)
            last_plane_size += c_plane.get_size()
            line_size.append(c_plane.get_stride())

        # create av frame with external buffer
        py_frame = VideoFrame(c_frame.get_width(), c_frame.get_height(),
                              c_frame.get_format(), buf_pointers, line_size)

        # set pts
        py_frame.pts = c_frame.get_pts()

        # set time base
        py_frame.time_base = Fraction(c_frame.get_time_base().num,
                                      c_frame.get_time_base().den)

        # transfer the buffer owner to av.frame
        c_frame.release()

        return py_frame

    def c_aframe_to_python_aframe(self, c_frame):
        buf_pointers = []
        line_size = []
        last_plane_size = 0

        # prepare buffer pointer and line size
        for c_plane in c_frame.py_get_planes():
            buf_pointers.append(c_plane.py_get_buffer())
            line_size.append(c_plane.get_size())

        # create av frame with external buffer
        py_frame = AudioFrame(c_frame.get_format(), c_frame.get_layout_name(),
                              c_frame.get_samples(), 1, buf_pointers,
                              line_size)
        # py_frame = AudioFrame(c_frame.get_width(), c_frame.get_height(),
        #                       c_frame.get_format(), buf_pointers, line_size)

        # set pts
        py_frame.pts = c_frame.get_pts()

        # set time base
        py_frame.time_base = Fraction(c_frame.get_time_base().num,
                                      c_frame.get_time_base().den)
        py_frame.sample_rate = c_frame.get_sample_rate()
        # py_frame.sample_rate =44100
        # transfer the buffer owner to av.frame
        c_frame.release()

        return py_frame

    def c_packet_to_python_packet(self, c_pkt):
        py_pkt = Packet()
        py_pkt.set_timestamp(c_pkt.get_timestamp())

        # for EOF packet, ignore data
        if c_pkt.get_timestamp() == Timestamp.EOF:
            return py_pkt

        c_data = c_pkt.py_get_data()

        # TODO: support other type as dict and so on
        if isinstance(c_data, csdk.VideoFrame):
            py_frame = self.c_vframe_to_python_vframe(c_data)
            py_pkt.set_data(py_frame)
            return py_pkt
        elif isinstance(c_data, csdk.AudioFrame):
            py_frame = self.c_aframe_to_python_aframe(c_data)
            py_pkt.set_data(py_frame)
            return py_pkt
        elif isinstance(c_data, numpy.ndarray):
            py_pkt.set_data(c_data)
            return py_pkt
        elif isinstance(c_data, dict):
            py_pkt.set_data(c_data)
            return py_pkt
        else:
            Log.log_node(LogLevel.ERROR, self.node_, "Unsupported data type:",
                         type(c_data))
            return None

    @staticmethod
    def to_str_array(l):
        r = []
        for value in l:
            r.append(str(value))
        return r

    def process(self, task=None):
        # create c task
        c_task = csdk.Task(c_module.to_str_array(task.get_inputs().keys()),
                           c_module.to_str_array(task.get_outputs().keys()))
        c_task.set_timestamp(task.get_timestamp())

        # convert python packet to c packet
        # and add to c task queue
        for (label, queue) in task.get_inputs().items():
            while not queue.empty():
                py_pkt = queue.get()
                c_pkt = self.python_packet_to_c_packet(py_pkt)
                if c_pkt is not None and c_pkt.defined():
                    c_task.add_packet_to_in_queue(str(label), c_pkt)
                    Log.log_node(LogLevel.ERROR, task.node_, "push frame",
                                 c_pkt.py_get_data())

        # call process of c module
        self.c_mod_.process(c_task)
        task.set_timestamp(c_task.get_timestamp())

        # get c output packets, convert to python packet
        # and add to python output queue
        for (label, queue) in task.get_outputs().items():
            while c_task.is_out_queue_empty(str(label)) == 0:
                c_pkt = csdk.Packet()
                c_task.pop_packet_from_out_queue(str(label), c_pkt)
                py_pkt = self.c_packet_to_python_packet(c_pkt)
                if py_pkt is not None and py_pkt.defined():
                    queue.put(py_pkt)
                    Log.log_node(LogLevel.ERROR, task.node_, "pull frame",
                                 py_pkt.get_data())

        return ProcessResult.OK
