
#include <bmf/sdk/hmp_import.h>
#include <bmf/sdk/sdk_interface.h>
#ifdef HMP_ENABLE_CUDA
#include <hmp/cuda/event.h>
#include <hmp/cuda/stream.h>
#endif

namespace bmf_sdk{


void OpaqueDataSet::set_private_data(int key, const OpaqueData &data)
{
    HMP_REQUIRE(key < OpaqueDataKey::kNumKeys, 
        "Private key {} is out of range, need less than {}", key, OpaqueDataKey::kNumKeys);

    opaque_set_[key] = data;
}

const OpaqueData& OpaqueDataSet::private_data(int key) const
{
    HMP_REQUIRE(key < OpaqueDataKey::kNumKeys, 
        "Private key {} is out of range, need less than {}", key, OpaqueDataKey::kNumKeys);
    return opaque_set_[key];
}


void OpaqueDataSet::private_merge(const OpaqueDataSet &from)
{
    for(int i = 0; i < OpaqueDataKey::kNumKeys; ++i){
        if(from.opaque_set_[i]){
            opaque_set_[i] = from.opaque_set_[i];
        }
    }
}


OpaqueDataSet& OpaqueDataSet::copy_props(const OpaqueDataSet& from)
{
    private_merge(from);
    return *this;
}


SequenceData& SequenceData::copy_props(const SequenceData& from)
{
    set_time_base(from.time_base());
    set_pts(from.pts());
    return *this;
}


struct Future::Private{
    uint64_t stream = 0;
#ifdef HMP_ENABLE_CUDA
    hmp::cuda::Event event;
#endif
};


Future::Future()
{
    self = std::make_shared<Private>();
}


void Future::set_stream(uint64_t stream)
{
    self->stream = stream;
}


uint64_t Future::stream() const
{
    return self->stream;
}

bool Future::ready() const
{
    auto d = device();
    if(d.type() == kCUDA || d.type() == kCPU){
#ifdef HMP_ENABLE_CUDA
        return !self->event.is_created() || self->event.query();
#else
        return true;
#endif
    }
    else{
        HMP_REQUIRE(false,
            "Future::ready: Not Implemented for device {}", d);
    }
}


void Future::record(bool use_current)
{
    auto d = device();
#ifdef HMP_ENABLE_CUDA
    if(d.type() == kCUDA || d.type() == kCPU){
        DeviceGuard guard(d);
        auto stream = use_current ? hmp::current_stream(kCUDA)
                                  : hmp::cuda::wrap_stream(self->stream, false);
        self->event.record(stream);
        if(use_current){
            self->stream = stream->handle();
        }
    }
#endif
}


void Future::synchronize()
{
    auto d = device();
#ifdef HMP_ENABLE_CUDA
    if(d.type() == kCUDA || d.type() == kCPU){
        if(self->event.is_created()){
            self->event.synchronize();
        }
        else{
            DeviceGuard guard(d);
            auto stream = hmp::cuda::wrap_stream(self->stream, false);
            stream.synchronize();
        }
    }
#endif
}

Future& Future::copy_props(const Future& from)
{
    set_stream(from.stream());
    return *this;
}

} //namespace bmf_sdk