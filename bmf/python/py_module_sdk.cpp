#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <map>
#include <bmf/sdk/module_functor.h>
#include <bmf/sdk/module_manager.h>
#include "py_type_cast.h"
#include <bmf/sdk/module.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/log_buffer.h>
#include <bmf/sdk/video_frame.h>
#include <bmf/sdk/audio_frame.h>
#include <bmf/sdk/bmf_av_packet.h>

namespace py = pybind11;


namespace bmf_sdk{

class PythonObject
{
    py::object obj_;
public:
    PythonObject(py::object &obj) : obj_(obj) {}

    py::object &obj() { return obj_; }

    ~PythonObject()
    {
        py::gil_scoped_acquire gil;
        obj_ = py::object();
    }
};

} //namespace bmf_sdk

BMF_DEFINE_TYPE(bmf_sdk::PythonObject)


namespace {


// packet casters for python
struct PacketCaster{
    std::function<py::object(bmf_sdk::Packet &)> from;
    std::function<bmf_sdk::Packet(py::object &)> to;
};

static std::map<std::string, PacketCaster> s_packet_casters;

std::string get_py_class_name(const py::object &cls)
{
    return cls.attr("__module__").cast<std::string>() + "." +
        cls.attr("__name__").cast<std::string>();
}

void register_packet_caster(const std::string &cls_name, const PacketCaster &caster)
{
    if(s_packet_casters.find(cls_name) != s_packet_casters.end()){
        throw std::runtime_error(cls_name + " has already registered");
    }

    s_packet_casters[cls_name] = caster;
}

void register_packet_caster(const py::object &cls, const PacketCaster &caster)
{
    auto cls_name = get_py_class_name(cls);
    register_packet_caster(cls_name, caster);
}


py::object packet_get(bmf_sdk::Packet &pkt, const py::object &cls)
{
    std::string cls_name;
    if(cls.is_none()){
        cls_name = "_bmf.sdk.PythonObject";
    }
    else{
        cls_name = get_py_class_name(cls);
        if(s_packet_casters.find(cls_name) == s_packet_casters.end()){
            cls_name = "_bmf.sdk.PythonObject";
        }
    }

    auto obj = s_packet_casters.at(cls_name).from(pkt);
    if(cls_name == "_bmf.sdk.PythonObject"){
        obj = obj.cast<bmf_sdk::PythonObject>().obj();
        if(cls.is_none() || obj.get_type().is(cls)){
            return obj;
        }
        else{
            throw std::runtime_error("class type is not matched");
        }
    }
    else{
        return obj;
    }
}


bool packet_is(bmf_sdk::Packet &pkt, const py::object &cls)
{
    try{
        if(pkt){
            packet_get(pkt, cls);
            return true;
        }
        else{
            return false;
        }
    }
    catch(...){
        return false;
    }
}

bmf_sdk::Packet packet_init(py::object &obj)
{
    if(obj.is_none()){
        return bmf_sdk::Packet();
    }

    auto cls_name = get_py_class_name(obj.attr("__class__"));
    if(s_packet_casters.find(cls_name) == s_packet_casters.end()){
        return bmf_sdk::Packet(bmf_sdk::PythonObject(obj)); //support python types
    }
    else{
        return s_packet_casters.at(cls_name).to(obj);
    }
}


#define PACKET_REGISTER_TYPE(name, T)         \
    register_packet_caster(name,      \
        PacketCaster{                           \
            .from = [](Packet &pkt){            \
                return py::cast(pkt.get<T>());  \
            },                                  \
            .to = [](py::object &obj){          \
                return Packet(obj.cast<T>());   \
            }                                   \
        });



#define PACKET_REGISTER_BMF_SDK_TYPE(T)         \
    PACKET_REGISTER_TYPE("_bmf.sdk." #T, T)



hmp::TensorOptions parse_tensor_options(const py::kwargs &kwargs, const hmp::TensorOptions &ref)
{
    using namespace hmp;

    TensorOptions opts(ref);

    if (kwargs.contains("pinned_memory")){
      opts = opts.pinned_memory(py::cast<bool>(kwargs["pinned_memory"]));
    }

    if (kwargs.contains("device")){
      auto device = kwargs["device"];
      if (PyUnicode_Check(device.ptr()))
        opts = opts.device(py::cast<std::string>(kwargs["device"]));
      else if (py::isinstance<Device>(kwargs["device"]))
        opts = opts.device(py::cast<Device>(kwargs["device"]));
      else
        opts = opts.device(py::cast<DeviceType>(kwargs["device"]));
    }

    if (kwargs.contains("dtype")){
      opts = opts.dtype(py::cast<ScalarType>(kwargs["dtype"]));
    }

    return opts;
}



struct PyPacketQueue
{
    PyPacketQueue(const std::shared_ptr<std::queue<bmf_sdk::Packet>> &queue)
        : q_(queue)
    {
    }

    bool empty() const { return q_->empty(); }

    int put(const bmf_sdk::Packet &pkt) { q_->push(pkt); return 0; }

    bmf_sdk::Packet get() 
    {
        auto pkt = q_->front();
        q_->pop();
        return pkt; 
    }

    bmf_sdk::Packet &front()
    {
        return q_->front();
    }

    std::shared_ptr<std::queue<bmf_sdk::Packet>> q_;
};


} //namespace



void module_sdk_bind(py::module &m)
{
    using namespace bmf_sdk;

    // Rational
    py::class_<Rational>(m, "Rational")
        .def(py::init<int, int>(), py::arg("num"), py::arg("den"))
        .def_readwrite("den", &Rational::den)
        .def_readwrite("num", &Rational::num)
    ;

    // Timestamp
    py::enum_<Timestamp>(m, "Timestamp")
        .value("kUNSET", Timestamp::UNSET)
        .value("kBMF_PAUSE", Timestamp::BMF_PAUSE)
        .value("kDYN_EOS", Timestamp::DYN_EOS)
        .value("kBMF_EOF", Timestamp::BMF_EOF)
        .value("kEOS", Timestamp::EOS)
        .value("kINF_SRC", Timestamp::INF_SRC)
        .value("kDONE", Timestamp::DONE)
        .export_values()
    ;


    // OpaqueDataSet
    py::enum_<OpaqueDataKey::Key>(m, "OpaqueDataKey")
        .value("kAVFrame", OpaqueDataKey::kAVFrame)
        .value("kAVPacket", OpaqueDataKey::kAVPacket)
        .value("kJsonParam", OpaqueDataKey::kJsonParam)
        .value("kReserved_3", OpaqueDataKey::kReserved_3)
        .value("kReserved_4", OpaqueDataKey::kReserved_4)
        .value("kReserved_5", OpaqueDataKey::kReserved_5)
        .value("kReserved_6", OpaqueDataKey::kReserved_6)
        .value("kReserved_7", OpaqueDataKey::kReserved_7)
        .export_values();

    py::class_<OpaqueDataSet>(m, "OpaqueDataSet")
        .def("private_merge", &OpaqueDataSet::private_merge, py::arg("from"))
        .def("private_get", [](OpaqueDataSet &self, py::object &cls) -> py::object{
            auto cls_name = get_py_class_name(cls);
            if(cls_name == "builtins.dict"){
                auto json_sptr = self.private_get<JsonParam>();
                if(!json_sptr){
                    return py::none();
                }
                return py::cast(*json_sptr);
            }
            else{
                throw std::invalid_argument(fmt::format("unsupported type {}", cls_name));
            }
        })
        .def("private_attach", [](OpaqueDataSet &self, py::object &obj){
            auto cls_name = get_py_class_name(obj.attr("__class__"));
            if(cls_name == "builtins.dict"){
                auto json = obj.cast<JsonParam>();
                self.private_attach<JsonParam>(&json);
            }
            else{
                throw std::invalid_argument(fmt::format("private attach type {} failed", cls_name));
            }
        })
        .def("copy_props", &OpaqueDataSet::copy_props, py::arg("from"))
    ;

    // SequenceData
    py::class_<SequenceData>(m, "SequenceData")
        .def_property("pts", &SequenceData::pts, &SequenceData::set_pts)
        .def_property("time_base", &SequenceData::time_base, &SequenceData::set_time_base)
        .def("copy_props", &SequenceData::copy_props, py::arg("from"))
    ;

    // Future
    py::class_<Future>(m, "Future")
        .def_property_readonly("device", &Future::device)
        .def_property("stream", &Future::stream, &Future::set_stream)
        .def("ready", &Future::ready)
        .def("record", &Future::record, py::arg("use_current")=true)
        .def("synchronize", &Future::synchronize)
        .def("copy_props", &Future::copy_props, py::arg("from"))
    ;

    // VideoFrame
    py::class_<VideoFrame, OpaqueDataSet, SequenceData, Future>(m, "VideoFrame")
        .def(py::init<Frame>())
        .def(py::init<Image>())
        .def(py::init([](int width, int height, const PixelInfo &pix_info, py::kwargs kwargs){
            auto opts = parse_tensor_options(kwargs, kUInt8);
            return VideoFrame(width, height, pix_info, opts.device());
        }), py::arg("width"), py::arg("height"), py::arg("pix_info"))
        .def(py::init([](int width, int height, int channels, ChannelFormat format, py::kwargs kwargs){
            auto opts = parse_tensor_options(kwargs, kUInt8);
            return VideoFrame(width, height, channels, format, opts);
        }), py::arg("width"), py::arg("height"), py::arg("channels")=3, py::arg("format")=kNCHW)
        .def("defined", &VideoFrame::operator bool)
        .def_property_readonly("width", &VideoFrame::width)
        .def_property_readonly("height", &VideoFrame::height)
        .def_property_readonly("dtype", &VideoFrame::dtype)
        .def("is_image", &VideoFrame::is_image)
        .def("image", &VideoFrame::image)
        .def("frame", &VideoFrame::frame)
        .def("to_image", &VideoFrame::to_image, py::arg("format")=kNCHW, py::arg("contiguous")=true)
        .def("to_frame", &VideoFrame::to_frame, py::arg("format"))
        .def("crop", &VideoFrame::crop, py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"))
        .def("cpu", &VideoFrame::cpu, py::arg("non_blocking")=false)
        .def("cuda", &VideoFrame::cuda)
        .def("copy_", &VideoFrame::copy_)
        .def("to", (VideoFrame(VideoFrame::*)(const Device&, bool) const)&VideoFrame::to, py::arg("device"), py::arg("non_blocking")=false)
        .def("to", (VideoFrame(VideoFrame::*)(ScalarType) const)&VideoFrame::to, py::arg("dtype"))
        .def("copy_props", &VideoFrame::copy_props, py::arg("from"))
    ;
    PACKET_REGISTER_BMF_SDK_TYPE(VideoFrame)

    // AudioFrame
    py::enum_<AudioChannelLayout::Layout>(m, "AudioChannelLayout")
        .value("kLAYOUT_MONO", AudioChannelLayout::kLAYOUT_MONO)
        .value("kLAYOUT_STEREO", AudioChannelLayout::kLAYOUT_STEREO)
        .value("kLAYOUT_2POINT1", AudioChannelLayout::kLAYOUT_2POINT1)
        .value("kLAYOUT_2_1", AudioChannelLayout::kLAYOUT_2_1)
        .value("kLAYOUT_SURROUND", AudioChannelLayout::kLAYOUT_SURROUND)
        .value("kLAYOUT_3POINT1", AudioChannelLayout::kLAYOUT_3POINT1)
        .value("kLAYOUT_4POINT0", AudioChannelLayout::kLAYOUT_4POINT0)
        .value("kLAYOUT_4POINT1", AudioChannelLayout::kLAYOUT_4POINT1)
        .value("kLAYOUT_2_2", AudioChannelLayout::kLAYOUT_2_2)
        .value("kLAYOUT_QUAD", AudioChannelLayout::kLAYOUT_QUAD)
        .value("kLAYOUT_5POINT0", AudioChannelLayout::kLAYOUT_5POINT0)
        .value("kLAYOUT_5POINT1", AudioChannelLayout::kLAYOUT_5POINT1)
        .value("kLAYOUT_5POINT0_BACK", AudioChannelLayout::kLAYOUT_5POINT0_BACK)
        .value("kLAYOUT_5POINT1_BACK", AudioChannelLayout::kLAYOUT_5POINT1_BACK)
        .value("kLAYOUT_6POINT0", AudioChannelLayout::kLAYOUT_6POINT0)
        .value("kLAYOUT_6POINT0_FRONT", AudioChannelLayout::kLAYOUT_6POINT0_FRONT)
        .value("kLAYOUT_HEXAGONAL", AudioChannelLayout::kLAYOUT_HEXAGONAL)
        .value("kLAYOUT_6POINT1", AudioChannelLayout::kLAYOUT_6POINT1)
        .value("kLAYOUT_6POINT1_BACK", AudioChannelLayout::kLAYOUT_6POINT1_BACK)
        .value("kLAYOUT_6POINT1_FRONT", AudioChannelLayout::kLAYOUT_6POINT1_FRONT)
        .value("kLAYOUT_7POINT0", AudioChannelLayout::kLAYOUT_7POINT0)
        .value("kLAYOUT_7POINT0_FRONT", AudioChannelLayout::kLAYOUT_7POINT0_FRONT)
        .value("kLAYOUT_7POINT1", AudioChannelLayout::kLAYOUT_7POINT1)
        .value("kLAYOUT_7POINT1_WIDE", AudioChannelLayout::kLAYOUT_7POINT1_WIDE)
        .value("kLAYOUT_7POINT1_WIDE_BACK", AudioChannelLayout::kLAYOUT_7POINT1_WIDE_BACK)
        .value("kLAYOUT_OCTAGONAL", AudioChannelLayout::kLAYOUT_OCTAGONAL)
        .value("kLAYOUT_HEXADECAGONAL", AudioChannelLayout::kLAYOUT_HEXADECAGONAL)
        .value("kLAYOUT_STEREO_DOWNMIX", AudioChannelLayout::kLAYOUT_STEREO_DOWNMIX)
        .export_values()
    ;


    py::class_<AudioFrame, OpaqueDataSet, SequenceData>(m, "AudioFrame")
        .def(py::init([](int samples, uint64_t layout, bool planer, py::kwargs kwargs){
            auto opts = parse_tensor_options(kwargs, kUInt8);
            return AudioFrame(samples, layout, planer, opts);
        }), py::arg("samples"), py::arg("layout"), py::arg("planer") = true)
        .def(py::init<TensorList, uint64_t, bool>(), py::arg("data"), py::arg("layout"), py::arg("planer")=true)
        .def("defined", &AudioFrame::operator bool)
        .def_property_readonly("layout", &AudioFrame::layout)
        .def_property_readonly("dtype", &AudioFrame::dtype)
        .def_property_readonly("planer", &AudioFrame::planer)
        .def_property_readonly("nsamples", &AudioFrame::nsamples)
        .def_property_readonly("nchannels", &AudioFrame::nchannels)
        .def_property_readonly("nplanes", &AudioFrame::nplanes)
        .def_property_readonly("planes", &AudioFrame::planes)
        .def_property("sample_rate", &AudioFrame::sample_rate, &AudioFrame::set_sample_rate)
        .def("copy_props", &AudioFrame::copy_props)
        .def("plane", &AudioFrame::plane)
        .def("clone", &AudioFrame::clone)
    ;
    PACKET_REGISTER_BMF_SDK_TYPE(AudioFrame)

    // BMFAVPacket
    py::class_<BMFAVPacket, OpaqueDataSet, SequenceData>(m, "BMFAVPacket")
        .def(py::init<>())
        .def(py::init<const Tensor&>())
        .def(py::init([](int size, const py::kwargs &kwargs){
            auto opts = parse_tensor_options(kwargs, kUInt8);
            return BMFAVPacket(size, opts);
        }))
        .def("defined", &BMFAVPacket::operator bool)
        .def_property_readonly("data", (Tensor&(BMFAVPacket::*)())&BMFAVPacket::data)
        .def_property_readonly("nbytes", &BMFAVPacket::nbytes)
        .def("copy_props", &BMFAVPacket::copy_props)
        .def("get_offset", &BMFAVPacket::get_offset)
        .def("get_whence", &BMFAVPacket::get_whence)
    ;
    PACKET_REGISTER_BMF_SDK_TYPE(BMFAVPacket)

    // PythonObject
    py::class_<PythonObject>(m, "PythonObject"); //dummy object
    PACKET_REGISTER_BMF_SDK_TYPE(PythonObject)

    // Packet
    py::class_<Packet>(m, "Packet")
        .def(py::init([](py::object &obj){
            return packet_init(obj);
        }))
        .def("defined", &Packet::operator bool)
        .def("is_", [](Packet &self, py::object& cls){
            return packet_is(self, cls);
        })
        .def("get", [](Packet &self, py::object& cls){
            return packet_get(self, cls);
        })
        .def_property("timestamp", &Packet::timestamp, &Packet::set_timestamp)
        .def_property_readonly("class_name", [](const Packet &pkt){
            return pkt.type_info().name;
        })
        .def_static("generate_eos_packet", &Packet::generate_eos_packet)
        .def_static("generate_eof_packet", &Packet::generate_eof_packet)
    ;
    //Map dict to JsonParam
    PACKET_REGISTER_TYPE("builtins.dict", JsonParam);
    PACKET_REGISTER_TYPE("builtins.str", std::string);

    // Task
    py::class_<PyPacketQueue>(m, "PacketQueue")
        .def("put", &PyPacketQueue::put)
        .def("get", &PyPacketQueue::get)
        .def("empty", &PyPacketQueue::empty)
        .def("front", &PyPacketQueue::front)
    ;

    auto convert_packet_queue_map = [](const PacketQueueMap &qmap){
        py::dict result;
        for(auto &qp : qmap){
            result[py::cast(qp.first)] = py::cast(PyPacketQueue(qp.second));
        }
        return result;
    };

    py::class_<Task>(m, "Task")
        .def(py::init<int, std::vector<int>, std::vector<int>>(),
            py::arg("node_id") = -1, 
            py::arg("input_stream_id_list") = std::vector<int>{},
            py::arg("input_stream_id_list") = std::vector<int>{})
        .def_property("timestamp", &Task::timestamp, &Task::set_timestamp)
        .def("fill_input_packet", &Task::fill_input_packet,
            py::arg("stream_id"), py::arg("packet"))
        .def("fill_output_packet", &Task::fill_output_packet,
            py::arg("stream_id"), py::arg("packet"))
        .def("pop_packet_from_out_queue", &Task::pop_packet_from_out_queue,
            py::arg("stream_id"), py::arg("packet"))
        .def("pop_packet_from_out_queue", [](Task &task, int stream_id){
            Packet pkt;
            if(!task.pop_packet_from_out_queue(stream_id, pkt)){
                throw std::runtime_error(fmt::format("Pop packet from output stream {} failed", stream_id));
            }
            return pkt;
        }, py::arg("stream_id"))
        .def("pop_packet_from_input_queue", &Task::pop_packet_from_input_queue,
            py::arg("stream_id"), py::arg("packet"))
        .def("pop_packet_from_input_queue", [](Task &task, int stream_id){
            Packet pkt;
            if(!task.pop_packet_from_input_queue(stream_id, pkt)){
                throw std::runtime_error(fmt::format("Pop packet from input stream {} failed", stream_id));
            }
            return pkt;
        }, py::arg("stream_id"))
        .def("output_queue_empty", &Task::output_queue_empty,
            py::arg("stream_id"))
        .def("input_queue_empty", &Task::input_queue_empty,
            py::arg("stream_id"))
        .def("get_outputs", [=](Task &task){
            return convert_packet_queue_map(task.get_outputs());
        })
        .def("get_inputs", [=](Task &task){
            return convert_packet_queue_map(task.get_inputs());
        })
        .def("get_input_stream_ids", &Task::get_input_stream_ids)
        .def("get_output_stream_ids", &Task::get_output_stream_ids)
        .def("get_node", &Task::get_node)
        .def("init", &Task::init,
            py::arg("node_id") = -1, 
            py::arg("input_stream_id_list") = std::vector<int>{},
            py::arg("input_stream_id_list") = std::vector<int>{})
    ;

    // AVLogBuffer
    py::class_<LogBuffer, std::unique_ptr<LogBuffer>>(m, "LogBuffer")
        .def(py::init([](py::list buffer, const std::string &level){
            return std::make_unique<LogBuffer>([=](const std::string &log){
                py::gil_scoped_acquire gil;
                buffer.append(py::cast(log));
            }, level);
        }))
        .def("close", &LogBuffer::close)
    ;


    // ModuleFunctor
    py::register_exception<ProcessDone>(m, "ProcessDone");

    py::class_<ModuleFunctor>(m, "ModuleFunctor")
        .def(py::init([](
                         int ninputs, int noutputs,
                         const std::string &name, 
                         const std::string &type,
                         const std::string &path,
                         const std::string &entry,
                         py::dict option, int node_id){
            auto &M = ModuleManager::instance();
            auto factory = M.load_module(name, type, path, entry);
            if(factory == nullptr){
                throw std::runtime_error("Create module " + name + " failed");
            }
            return ModuleFunctor(
                factory->make(node_id, py::cast<JsonParam>(option)),
                ninputs, noutputs);
        }), py::arg("ninputs"), py::arg("noutputs"),
            py::arg("name"), py::arg("type")=std::string(),
            py::arg("path")=std::string(), py::arg("entry")=std::string(), 
            py::arg("option")=py::dict(), py::arg("node_id")=0)

        .def_nogil("__call__", &ModuleFunctor::operator(), py::arg("inputs"))
        .def_nogil("execute", &ModuleFunctor::execute, py::arg("inputs"), py::arg("cleanup")=true)
        .def_nogil("fetch", &ModuleFunctor::fetch, py::arg("port"))
    ;


} //
