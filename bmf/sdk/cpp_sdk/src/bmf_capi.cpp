#include <bmf/sdk/bmf_capi.h>
#include <hmp/tensor.h>
#include <hmp/format.h>


#define BMF_PROTECT(...) 	            \
	try{                                \
        __VA_ARGS__                     \
    } catch(const std::exception &e){   \
        s_bmf_last_error = e.what();    \
    }


using namespace bmf_sdk;


thread_local std::string s_bmf_last_error;

void bmf_set_last_error(const char *errstr)
{
    s_bmf_last_error = errstr;
}

const char *bmf_last_error()
{
    return s_bmf_last_error.c_str();
}


const char *bmf_sdk_version()
{
    return BMF_SDK_VERSION;
}


char *bmf_strdup(char const *src)
{
    char *res;
#ifdef _POSIX_VERSION
    res = strdup(src);
#else
    res = (char *) malloc(strlen(src) + 1);
    strcpy(res, src);
#endif
    return res;
}

////////// Common ////////////
char* bmf_json_param_dump(bmf_JsonParam json)
{
    auto str = json->dump();
    return strdup(str.c_str());
}

bmf_JsonParam bmf_json_param_parse(const char* str)
{
    BMF_PROTECT(
        bmf_JsonParam j = new JsonParam();
        j->parse(str);
        return j;
    )
    return nullptr;
}

void bmf_json_param_free(bmf_JsonParam json)
{
    if(json){
        delete json;
    }
}


////////// VideoFrame ////////////

bmf_VideoFrame bmf_vf_from_image(hmp_Image image)
{
    BMF_PROTECT(
	    return new VideoFrame(*image);
    )

    return nullptr;
}

bmf_VideoFrame bmf_vf_from_frame(hmp_Frame frame)
{
    BMF_PROTECT(
        return new VideoFrame(*frame);
    )

    return nullptr;
}

bmf_VideoFrame bmf_vf_make_frame(int width, int height, 
				const hmp_PixelInfo pix_info, const char *device)
{
    BMF_PROTECT(
        return new VideoFrame(width, height, *pix_info, Device(device));
    )
    return nullptr;
}

bmf_VideoFrame bmf_vf_make_image(int width, int height, int channels,
				int format, int dtype, const char *device,
				bool pinned_memory)
{
    BMF_PROTECT(
        return new VideoFrame(width, height, channels,
                 (ChannelFormat)format,
                 TensorOptions((ScalarType)dtype)
                    .device(Device(device)).pinned_memory(pinned_memory));
    )
    return nullptr;
}


void bmf_vf_free(bmf_VideoFrame vf)
{
    if(vf){
        delete vf;
    }
}


bool bmf_vf_defined(const bmf_VideoFrame vf)
{
    return *vf;
}
int bmf_vf_width(const bmf_VideoFrame vf)
{
    return vf->width();
}

int bmf_vf_height(const bmf_VideoFrame vf)
{
    return vf->height();
}

int bmf_vf_dtype(const bmf_VideoFrame vf)
{
    return (int)vf->dtype();
}

bool bmf_vf_is_image(const bmf_VideoFrame vf)
{
    return vf->is_image();
}
const hmp_Image bmf_vf_image(const bmf_VideoFrame vf)
{
    BMF_PROTECT(
        return (const hmp_Image)&vf->image();
    )

    return nullptr;
}

const hmp_Frame bmf_vf_frame(const bmf_VideoFrame vf)
{
    BMF_PROTECT(
        return (const hmp_Frame)&vf->frame();
    )

    return nullptr;
}


bmf_VideoFrame bmf_vf_to_image(const bmf_VideoFrame vf, int format, bool contiguous)
{
    BMF_PROTECT(
        auto tmp = vf->to_image((ChannelFormat)format, contiguous);
        return new VideoFrame(tmp);
    )
    return nullptr;
}

bmf_VideoFrame bmf_vf_to_frame(const bmf_VideoFrame vf, const hmp_PixelInfo pix_info)
{
    BMF_PROTECT(
        auto tmp = vf->to_frame(*pix_info);
        return new VideoFrame(tmp);
    )

    return nullptr;
}


bmf_VideoFrame bmf_vf_cpu(const bmf_VideoFrame vf, bool non_blocking)
{
    BMF_PROTECT(
        auto tmp = vf->cpu(non_blocking);
        return new VideoFrame(tmp);
    )
    return nullptr;
}

bmf_VideoFrame bmf_vf_cuda(const bmf_VideoFrame vf)
{
    BMF_PROTECT(
        auto tmp = vf->cuda();
        return new VideoFrame(tmp);
    )
    return nullptr;
}


int bmf_vf_device_type(const bmf_VideoFrame vf)
{
    return (int)vf->device().type();
}

int bmf_vf_device_index(const bmf_VideoFrame vf)
{
    return vf->device().index();
}

void bmf_vf_copy_from(bmf_VideoFrame vf, const bmf_VideoFrame from)
{
    vf->copy_(*from);
}

bmf_VideoFrame bmf_vf_to_device(const bmf_VideoFrame vf, const char *device, bool non_blocking)
{
    BMF_PROTECT(
        return new VideoFrame(vf->to(Device(device), non_blocking));
    )
    return nullptr;
}

bmf_VideoFrame bmf_vf_to_dtype(const bmf_VideoFrame vf, int dtype)
{
    BMF_PROTECT(
        return new VideoFrame(vf->to((ScalarType)dtype));
    )
    return nullptr;
}

void bmf_vf_copy_props(bmf_VideoFrame vf, const bmf_VideoFrame from)
{
    vf->copy_props(*from);
}

void bmf_vf_private_merge(bmf_VideoFrame vf, const bmf_VideoFrame from)
{
    vf->private_merge(*from);
}

const bmf_JsonParam bmf_vf_private_get_json_param(const bmf_VideoFrame vf)
{
    return (const bmf_JsonParam)vf->private_get<JsonParam>();
}

void bmf_vf_private_attach_json_param(bmf_VideoFrame vf, const bmf_JsonParam json_param)
{
    vf->private_attach<JsonParam>(json_param);
}

void bmf_vf_set_pts(bmf_VideoFrame vf, int64_t pts)
{
    vf->set_pts(pts);
}

int64_t bmf_vf_pts(bmf_VideoFrame vf)
{
    return vf->pts();
}

void bmf_vf_set_time_base(bmf_VideoFrame vf, int num, int den)
{
    vf->set_time_base(Rational(num, den));
}

void bmf_vf_time_base(bmf_VideoFrame vf, int *num, int *den)
{
    *num = vf->time_base().num;
    *den = vf->time_base().den;
}

bool bmf_vf_ready(const bmf_VideoFrame vf)
{
    return vf->ready();
}

void bmf_vf_record(bmf_VideoFrame vf, bool use_current)
{
    vf->record(use_current);
}

void bmf_vf_synchronize(bmf_VideoFrame vf)
{
    vf->synchronize();
}


////////// AudioFrame ////////////
bmf_AudioFrame bmf_af_make_from_data(hmp_Tensor *data, int size, uint64_t layout, bool planer)
{
    BMF_PROTECT(
        std::vector<Tensor> tensor_list;
        for (int i=0; i<size; i++){
            tensor_list.push_back(*data[i]);
        }
        return new AudioFrame(tensor_list, layout, planer);
    )
    return nullptr;
}

bmf_AudioFrame bmf_af_make(int samples, uint64_t layout, bool planer, int dtype)
{
    BMF_PROTECT(
        auto tensor_options = hmp::TensorOptions((hmp::ScalarType)dtype);
        return new AudioFrame(samples, layout, planer, tensor_options);
    )
    return nullptr;
}

void bmf_af_free(bmf_AudioFrame af)
{
    if(af){
        delete af;
    }
}

bool bmf_af_defined(const bmf_AudioFrame af)
{
    return *af;
}

uint64_t bmf_af_layout(const bmf_AudioFrame af)
{
    return af->layout();
}

int bmf_af_dtype(const bmf_AudioFrame af)
{
    return (int)af->dtype();
}

bool bmf_af_planer(const bmf_AudioFrame af)
{
    return af->planer();
}

int bmf_af_nsamples(const bmf_AudioFrame af)
{
    return af->nsamples();
}

int bmf_af_nchannels(const bmf_AudioFrame af)
{
    return af->nchannels();
}

void bmf_af_set_sample_rate(const bmf_AudioFrame af, float sr)
{
    af->set_sample_rate(sr);
}

float bmf_af_sample_rate(const bmf_AudioFrame af)
{
    return af->sample_rate();
}

int bmf_af_planes(const bmf_AudioFrame af, hmp_Tensor *data)
{
    if(data != nullptr){
        for(size_t i = 0; i < af->planes().size(); ++i){
            data[i] = new Tensor(af->planes()[i]);
        }
    }
    return (int)af->planes().size();
}

int bmf_af_nplanes(const bmf_AudioFrame af)
{
    return af->nplanes();
}

hmp_Tensor bmf_af_plane(const bmf_AudioFrame af, int p)
{
    BMF_PROTECT(
        return new Tensor(af->plane(p));
    )
    return nullptr;
}

void bmf_af_copy_props(bmf_AudioFrame af, const bmf_AudioFrame from)
{
    af->copy_props(*from);
}

void bmf_af_private_merge(bmf_AudioFrame af, const bmf_AudioFrame from)
{
    af->private_merge(*from);
}

const bmf_JsonParam bmf_af_private_get_json_param(const bmf_AudioFrame af)
{
    return (const bmf_JsonParam)af->private_get<JsonParam>();
}

void bmf_af_private_attach_json_param(bmf_AudioFrame af, const bmf_JsonParam json_param)
{
    af->private_attach<JsonParam>(json_param);
}

void bmf_af_set_pts(bmf_AudioFrame af, int64_t pts)
{
    af->set_pts(pts);
}

int64_t bmf_af_pts(bmf_AudioFrame af)
{
    return af->pts();
}

void bmf_af_set_time_base(bmf_AudioFrame af, int num, int den)
{
    af->set_time_base(Rational(num, den));
}

void bmf_af_time_base(bmf_AudioFrame af, int *num, int *den)
{
    *num = af->time_base().num;
    *den = af->time_base().den;
}


////////// BMFAVPacket ////////////
bmf_BMFAVPacket bmf_pkt_make_from_data(hmp_Tensor data)
{
    BMF_PROTECT(
        return new BMFAVPacket(*data);
    )
    return nullptr;
}

bmf_BMFAVPacket bmf_pkt_make(int size, int dtype)
{
    BMF_PROTECT(
        auto tensor_options = hmp::TensorOptions((hmp::ScalarType)dtype);
        return new BMFAVPacket(size, tensor_options);
    )
    return nullptr;
}

void bmf_pkt_free(bmf_BMFAVPacket pkt)
{
    if(pkt){
        delete pkt;
    }
}

bool bmf_pkt_defined(const bmf_BMFAVPacket pkt)
{
    return *pkt;
}

hmp_Tensor bmf_pkt_data(const bmf_BMFAVPacket pkt)
{
    BMF_PROTECT(
        return new Tensor(pkt->data());
    )
    return nullptr;
}

void* bmf_pkt_data_ptr(bmf_BMFAVPacket pkt)
{
    return pkt->data_ptr();
}

const void* bmf_pkt_data_const_ptr(const bmf_BMFAVPacket pkt)
{
    return pkt->data_ptr();
}

int bmf_pkt_nbytes(const bmf_BMFAVPacket pkt)
{
    return pkt->nbytes();
}

void bmf_pkt_copy_props(bmf_BMFAVPacket pkt, const bmf_BMFAVPacket from)
{
    pkt->copy_props(*from);
}

void bmf_pkt_private_merge(bmf_BMFAVPacket pkt, const bmf_BMFAVPacket from)
{
    pkt->private_merge(*from);
}

const bmf_JsonParam bmf_pkt_private_get_json_param(const bmf_BMFAVPacket pkt)
{
    return (const bmf_JsonParam)pkt->private_get<JsonParam>();
}

void bmf_pkt_private_attach_json_param(bmf_BMFAVPacket pkt, const bmf_JsonParam json_param)
{
    pkt->private_attach<JsonParam>(json_param);
}

void bmf_pkt_set_pts(bmf_BMFAVPacket pkt, int64_t pts)
{
    pkt->set_pts(pts);
}

int64_t bmf_pkt_pts(bmf_BMFAVPacket pkt)
{
    return pkt->pts();
}

void bmf_pkt_set_time_base(bmf_BMFAVPacket pkt, int num, int den)
{
    pkt->set_time_base(Rational(num, den));
}

void bmf_pkt_time_base(bmf_BMFAVPacket pkt, int *num, int *den)
{
    *num = pkt->time_base().num;
    *den = pkt->time_base().den;
}


/////////////// TypeInfo /////////////////
const char* bmf_type_info_name(const bmf_TypeInfo type_info)
{
    return type_info->name;
}

unsigned long bmf_type_info_index(const bmf_TypeInfo type_info)
{
    return type_info->index;
}

//////////////// Packet //////////////////
void bmf_packet_free(bmf_Packet pkt)
{
    if(pkt){
        delete pkt;
    }
}

int bmf_packet_defined(bmf_Packet pkt)
{
    return pkt != nullptr && pkt->operator bool();
}

const bmf_TypeInfo bmf_packet_type_info(const bmf_Packet pkt)
{
    return (const bmf_TypeInfo)&pkt->type_info();
}

bmf_Packet bmf_packet_generate_eos_packet()
{
    return new Packet(Packet::generate_eos_packet());
}

bmf_Packet bmf_packet_generate_eof_packet()
{
    return new Packet(Packet::generate_eof_packet());
}

bmf_Packet bmf_packet_generate_empty_packet()
{
    return new Packet();
}

int64_t bmf_packet_timestamp(const bmf_Packet pkt)
{
    return pkt->timestamp();
}

void bmf_packet_set_timestamp(bmf_Packet pkt, int64_t timestamp)
{
    return pkt->set_timestamp(timestamp);
}

bmf_Packet bmf_packet_from_videoframe(const bmf_VideoFrame vf)
{
    BMF_PROTECT(
        return new Packet(*vf);
    )

    return nullptr;
}

bmf_VideoFrame bmf_packet_get_videoframe(const bmf_Packet pkt)
{
    BMF_PROTECT(
        return new VideoFrame(pkt->get<VideoFrame>());
    )
    return nullptr;
}

int bmf_packet_is_videoframe(const bmf_Packet pkt)
{
    return pkt->is<VideoFrame>();
}

bmf_Packet bmf_packet_from_audioframe(const bmf_AudioFrame af)
{
    BMF_PROTECT(
        return new Packet(*af);
    )

    return nullptr;
}

bmf_AudioFrame bmf_packet_get_audioframe(const bmf_Packet pkt)
{
    BMF_PROTECT(
        return new AudioFrame(pkt->get<AudioFrame>());
    )
    return nullptr;
}

int bmf_packet_is_audioframe(const bmf_Packet pkt)
{
    return pkt->is<AudioFrame>();
}

bmf_Packet bmf_packet_from_bmfavpacket(const bmf_BMFAVPacket bmf_av_pkt)
{
    BMF_PROTECT(
        return new Packet(*bmf_av_pkt);
    )

    return nullptr;
}

bmf_BMFAVPacket bmf_packet_get_bmfavpacket(const bmf_Packet pkt)
{
    BMF_PROTECT(
        return new BMFAVPacket(pkt->get<BMFAVPacket>());
    )
    return nullptr;
}

int bmf_packet_is_bmfavpacket(const bmf_Packet pkt)
{
    return pkt->is<BMFAVPacket>();
}


bmf_Packet bmf_packet_from_json_param(const bmf_JsonParam json)
{
    BMF_PROTECT(
        return new Packet(*json);
    )
    return nullptr;
}

bmf_JsonParam bmf_packet_get_json_param(const bmf_Packet pkt)
{
    BMF_PROTECT(
        return new JsonParam(pkt->get<JsonParam>());
    )
    return nullptr;

}

int bmf_packet_is_json_param(const bmf_Packet pkt)
{
    return pkt->is<JsonParam>();
}


//////////////////// Task /////////////////////
bmf_Task bmf_task_make(int node_id, int *istream_ids, int ninputs, int *ostream_ids, int noutputs)
{
    BMF_PROTECT(
        std::vector<int> iids(istream_ids, istream_ids + ninputs);
        std::vector<int> oids(ostream_ids, ostream_ids + noutputs);
        return new Task(node_id, iids, oids);
    )
    return nullptr;
}

void bmf_task_free(bmf_Task task)
{
    if(task){
        delete task;
    }
}

int bmf_task_fill_input_packet(bmf_Task task, int stream_id, const bmf_Packet packet)
{
    BMF_PROTECT(
        return task->fill_input_packet(stream_id, *packet);
    )
    return 0;
}

int bmf_task_fill_output_packet(bmf_Task task, int stream_id, const bmf_Packet packet)
{
    BMF_PROTECT(
        return task->fill_output_packet(stream_id, *packet);
    )
    return 0;
}

bmf_Packet bmf_task_pop_packet_from_out_queue(bmf_Task task, int stream_id)
{
    BMF_PROTECT(
        Packet pkt;
        if(task->pop_packet_from_out_queue(stream_id, pkt)){
            return new Packet(pkt);
        }
        else{
            throw std::runtime_error(
                fmt::format("stream id out of range or no packet to pop from output stream {}", stream_id));
        }
    )
    return nullptr;
}

bmf_Packet bmf_task_pop_packet_from_input_queue(bmf_Task task, int stream_id)
{
    BMF_PROTECT(
        Packet pkt;
        if(task->pop_packet_from_input_queue(stream_id, pkt)){
            return new Packet(pkt);
        }
        else{
            throw std::runtime_error(
                fmt::format("stream id out of range or no packet to pop from input stream {}", stream_id));
        }
    )
    return nullptr;
}

int64_t bmf_task_timestamp(const bmf_Task task)
{
    return task->timestamp();
}

void bmf_task_set_timestamp(const bmf_Task task, int64_t timestamp)
{
    return task->set_timestamp(timestamp);
}


int bmf_task_get_input_stream_ids(bmf_Task task, int *ids)
{
    BMF_PROTECT(
        auto sids = task->get_input_stream_ids();
        for(size_t i = 0; i < sids.size() && ids; ++i){
            ids[i] = sids[i];
        }
        return sids.size();
    )
    return -1;
}

int bmf_task_get_output_stream_ids(bmf_Task task, int *ids)
{
    BMF_PROTECT(
        auto sids = task->get_output_stream_ids();
        for(size_t i = 0; i < sids.size() && ids; ++i){
            ids[i] = sids[i];
        }
        return sids.size();
    )
    return -1;
}

int bmf_task_get_node(bmf_Task task)
{
    return task->get_node();
}



//////////////// ModuleFunctor ////////////
bmf_ModuleFunctor bmf_module_functor_make(
            const char* name, const char* type, const char* path, 
            const char *entry, const char* option,
            int ninputs, int noutputs, int node_id)
{
    BMF_PROTECT(
        auto &M = ModuleManager::instance();
        ModuleInfo info(name, type, entry, path);
        auto factory = M.load_module(info);
        if(factory == nullptr){
            throw std::runtime_error("Load module " + info.module_name + " failed");
        }

        JsonParam json_option;
        json_option.parse(option);
        auto m = factory->make(node_id, json_option);
        return new ModuleFunctor(m, ninputs, noutputs);
    )

    return nullptr;
}

void bmf_module_functor_free(bmf_ModuleFunctor mf)
{
    if(mf){
        delete mf;
    }
}

bmf_Packet* bmf_module_functor_call(bmf_ModuleFunctor mf, 
            const bmf_Packet *inputs, int ninputs, int *noutputs, bool *is_done)
{
    BMF_PROTECT(
        std::vector<Packet> ipkts;
        for(int i = 0; i < ninputs; ++i){
            if(inputs[i]){
                ipkts.push_back(*inputs[i]);
            }
            else{
                ipkts.push_back(Packet());
            }
        }

        std::vector<Packet> opkts;
        try{
            opkts = (*mf)(ipkts);
        }
        catch(ProcessDone &e){
            s_bmf_last_error = e.what();
            *is_done = true;
            return nullptr;
        }

        auto buf = (bmf_Packet*)malloc(opkts.size()*sizeof(bmf_Packet));
        for(size_t i = 0; i < opkts.size(); ++i){
            if(opkts[i]){
                buf[i] = new Packet(opkts[i]);
            }
            else{
                buf[i] = nullptr;
            }
        }

        if(noutputs){
            *noutputs = opkts.size();
        }
        return buf;
    )

    return nullptr;
}


int bmf_module_functor_execute(bmf_ModuleFunctor mf, const bmf_Packet *inputs, int ninputs, bool cleanup, bool *is_done)
{
    BMF_PROTECT(
        std::vector<Packet> ipkts;
        for(int i = 0; i < ninputs; ++i){
            if(inputs[i]){
                ipkts.push_back(*inputs[i]);
            }
            else{
                ipkts.push_back(Packet());
            }
        }

        try{
            mf->execute(ipkts, cleanup);
            return 0;
        }
        catch(ProcessDone &e){
            s_bmf_last_error = e.what();
            *is_done = true;
            return -1;
        }
    )

    return -1;

}

bmf_Packet* bmf_module_functor_fetch(bmf_ModuleFunctor mf, int index, int *noutputs, bool *is_done)
{
    BMF_PROTECT(
        auto opkts = mf->fetch(index);

        auto buf = (bmf_Packet*)malloc(opkts.size()*sizeof(bmf_Packet));
        for(size_t i = 0; i < opkts.size(); ++i){
            if(opkts[i]){
                buf[i] = new Packet(opkts[i]);
            }
            else{
                buf[i] = nullptr;
            }
        }

        if(noutputs){
            *noutputs = opkts.size();
        }
        return buf;
    )

    return nullptr;
}