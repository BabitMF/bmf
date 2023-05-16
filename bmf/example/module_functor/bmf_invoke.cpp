#include <iostream>
#include <fstream>
#include <filesystem>
//https://github.com/p-ranav/glob
#include <glob.hpp>
#include <bmf/sdk/module_functor.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/video_frame.h>

namespace fs = std::filesystem;

#define BMF_PROTECT(...) 	            \
	try{                                \
        __VA_ARGS__                     \
    } catch(const std::exception &e){   \
        fmt::print("Exception {}\n", e.what()); \
        return -1;                      \
    }

using namespace bmf_sdk;

namespace {


Packet load_from_file(const std::string &fn, 
                      const JsonParam &param)
{
    ScalarType dtype = kUInt8;
    auto width = param.get<int>("width");
    auto height = param.get<int>("height");
    auto ext = fn.substr(fn.find_last_of("."));

    auto data = hmp::fromfile(fn, dtype);
    if(ext == ".rgb"){
        data = data.reshape(SizeArray{height, width, 3});
        //return VideoFrame(Image(data, kNHWC));
        auto rgb = hmp::PixelInfo(hmp::PF_RGB24);
        return VideoFrame(hmp::Frame({data}, rgb));
    }
    else if(ext == ".yuv"){
        auto format_str = param.get<std::string>("format");
        PixelFormat pformat;
        if(format_str == "yuv420p"){
            pformat = PixelFormat::PF_YUV420P;
        }
        else if(format_str == "yuva420p"){
            pformat = PixelFormat::PF_YUVA420P;
        }
        else{
            throw std::runtime_error(fmt::format("Unsupport image/frame format {}", format_str));
        }

        hmp::PixelFormatDesc desc(pformat);
        if(desc.infer_nitems(width, height) != data.nitems()){
            throw std::runtime_error(fmt::format("Invalid image size"));
        }

        TensorList planes;
        int64_t off = 0;
        for(int i = 0; i < desc.nplanes(); ++i){
            auto n = desc.infer_nitems(width, height, i);
            auto w = desc.infer_width(width, i);
            auto h = desc.infer_height(height, i);
            auto plane = data.slice(0, off, off + n).reshape(SizeArray{h, w, -1});
            planes.push_back(plane);
            off += n;
        }

        auto frame = Frame(planes, width, height, pformat);
        return VideoFrame(frame);
    }
    else{
        throw std::runtime_error(fmt::format("Unsupport image/frame file format {}", ext));
    }
}


void store_to_file(const Packet &pkt, const std::string &fn)
{
    if(pkt.is<VideoFrame>()){
        auto vf = pkt.get<VideoFrame>();
        Tensor data;
        auto frame = vf.frame();
        int64_t nitems = 0;
        for(int i = 0; i < frame.nplanes(); ++i){
            nitems += frame.plane(i).nitems();
        }

        //concatenate all planes
        data = hmp::empty(SizeArray{nitems}, frame.plane(0).dtype());
        nitems = 0;
        for(int i = 0; i < frame.nplanes(); ++i){
            auto &plane = frame.plane(i);
            data.slice(0, nitems, nitems + plane.nitems()).copy_(plane.flatten());
            nitems += plane.nitems();
        }
        data.tofile(fn);
    }
    else if(pkt.is<JsonParam>()){
        auto json = pkt.get<JsonParam>();
        std::ofstream t(fn, std::ios::binary);
        t << json.json_value_;
    }
    else{
        throw std::runtime_error(fmt::format("Unsupported data type {}", pkt.type_info().name));
    }
}



class VFSource : public Module
{
    std::vector<Packet> pkts;
    int                 idx;
public:
    VFSource(int node_id,  JsonParam &option)
        : Module(node_id, option)
    {
        //auto type = option.get<std::string>("type");
        auto param = option["param"];

        for(auto &fn : glob::glob(option.get<std::string>("filter"))){
            pkts.push_back(
                load_from_file(fn.string(), param));
        }
        idx = 0;
        HMP_INF("VFSource load {} files", pkts.size());
    }

    int process(Task &task) override
    {
        if(idx < pkts.size()){
            pkts[idx].set_timestamp(idx);
            task.fill_output_packet(0, pkts[idx]);
            idx += 1;
        }
        else{
            task.set_timestamp(Timestamp::DONE);                        
        }
        return 0;
    }
};


class VFSink : public Module
{
    std::string fn_format; 
    int idx;
public:
    VFSink(int node_id, const JsonParam &option)
        : Module(node_id, option)
    {
        fn_format = option.get<std::string>("name");
        idx = 0;
    }

    ~VFSink()
    {
        HMP_INF("VFSink write {} files", idx);
    }

    int process(Task &task) override
    {
        Packet pkt;
        while(task.pop_packet_from_input_queue(0, pkt)){
            auto fn = fmt::format(fn_format, idx);
            store_to_file(pkt, fn);
            idx += 1;
        }

        return 0;
    }
};


REGISTER_MODULE_CLASS(VFSource)
REGISTER_MODULE_CLASS(VFSink)


} //namespace



int main(int argc, char *argv[])
{
    //
    auto info_fn = argc > 1 ? argv[1] : "../bmf/example/module_functor/invoke_module.json";
    if(!fs::exists(info_fn)){
        fmt::print("Config file {} not found", info_fn);
        return -1;
    }

    BMF_PROTECT(
        // load input & module & output info
        auto info = bmf_sdk::JsonParam();
        info.load(info_fn);

        ModuleFunctor input, module, output;

        //
        if(info.has_key("input")){
            auto input_info = info["input"];
            input = make_sync_func(
                ModuleInfo(input_info.get<std::string>("name"), "c++", ""),
                0, 1,
                input_info["option"]
            );
        }

        //
        if(info.has_key("module")){
            auto module_info = info["module"];
            auto name = module_info.get<std::string>("name");
            auto type = module_info.get<std::string>("type");
            auto path = module_info.get<std::string>("path");
            auto entry = module_info.get<std::string>("entry");
            auto option = module_info["option"];

            module = make_sync_func(
                ModuleInfo(name, type, entry, path),
                1, 1,
                option
            );
        }

        //
        if(info.has_key("output")){
            auto output_info = info["output"];
            output = make_sync_func(
                ModuleInfo(output_info.get<std::string>("name"), "c++", ""),
                1, 0,
                output_info["option"]
            );
        }

        //
        while(true){
            std::vector<Packet> inputs;
            std::vector<Packet> outputs;

            if(input.defined()){
                input.execute({}, true);
                inputs = input.fetch(0);
            }

            //FIXME handle eof packet
            if(module.defined()){
                module.execute(inputs, true);
                outputs = module.fetch(0);
            }
            else{
                outputs = inputs;
            }

            if(output.defined()){
                output.execute(outputs, true);
            }
        }
    ) //BMF_PROTECT

    return 0;
}
