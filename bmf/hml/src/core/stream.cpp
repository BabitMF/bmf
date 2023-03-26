/*
 * Copyright 2023 Babit Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <mutex>
#include <hmp/core/stream.h>

namespace hmp{

namespace {

HMP_DECLARE_STREAM_MANAGER(kCPU);
#ifdef HMP_ENABLE_CUDA
HMP_DECLARE_STREAM_MANAGER(kCUDA);
#endif


const static Device sCPUDevice(kCPU);

//dummy cpu stream
class CPUStream : public StreamInterface
{
public:
    CPUStream()
    {
    }

    CPUStream(uint64_t flags)
    {
    }

    ~CPUStream()
    {
    }

    const Device &device() const override
    {
        return sCPUDevice;
    }

    StreamHandle handle() const override
    {
        return 0;
    }

    bool query() override
    {
        return true;
    }

    virtual void synchronize() override
    {
    }
}; 


//
static thread_local RefPtr<CPUStream> sCurrentStream; 

class CPUStreamManager : public impl::StreamManager
{
public:
    void setCurrent(const Stream& stream) override
    {
        auto ref = stream.unsafeGet();
        auto cpuStream = dynamic_cast<CPUStream*>(ref.get());
        HMP_REQUIRE(cpuStream, "Invalid CPU stream");

        sCurrentStream = ref.cast<CPUStream>();
    }

    optional<Stream> getCurrent() const override
    {
        if(!sCurrentStream){
            return Stream(makeRefPtr<CPUStream>()); //get default stream by default
        }
        else{
            return Stream(sCurrentStream);
        }
    }

    Stream create(uint64_t flags = 0) override
    {
        return Stream(makeRefPtr<CPUStream>(flags));
    }

};

static CPUStreamManager sCPUStreamManager;

HMP_REGISTER_STREAM_MANAGER(kCPU, &sCPUStreamManager);

} //namespace



namespace impl{

static StreamManager *sStreamManagers[static_cast<int>(DeviceType::NumDeviceTypes)];

void registerStreamManager(DeviceType dtype, StreamManager *sm)
{
    //as it only init before main, so no lock is needed
    sStreamManagers[static_cast<int>(dtype)] = sm; 
}

} //namespace impl


std::string stringfy(const Stream &stream)
{
    return fmt::format("Stream({}, {})",
        stringfy(stream.device()), stream.handle());
}

optional<Stream> current_stream(DeviceType dtype)
{
    auto sm = impl::sStreamManagers[static_cast<int>(dtype)];
    HMP_REQUIRE(sm, "Stream on device type {} is not supported", dtype);
    return sm->getCurrent();
}

void set_current_stream(const Stream &stream)
{
    auto dtype = stream.device().type();
    auto sm = impl::sStreamManagers[static_cast<int>(dtype)];
    HMP_REQUIRE(sm, "Stream on device type {} is not supported", dtype);
    sm->setCurrent(stream);
}

Stream create_stream(DeviceType dtype, uint64_t flags)
{
#ifndef HMP_BUILD_SHARED
    HMP_IMPORT_STREAM_MANAGER(kCPU);
#ifdef HMP_ENABLE_CUDA
    HMP_IMPORT_STREAM_MANAGER(kCUDA);
#endif
#endif


    auto sm = impl::sStreamManagers[static_cast<int>(dtype)];
    HMP_REQUIRE(sm, "Stream on device type {} is not supported", dtype);
    return sm->create(flags);
}

StreamGuard::StreamGuard(const Stream& stream)
{
    auto dtype = stream.device().type();
    auto current = current_stream(dtype);
    if(current != stream){
        set_current_stream(stream);
    }
    origin_ = current;
}


StreamGuard::StreamGuard(StreamGuard &&other)
{
    origin_ = other.origin_;
    other.origin_.reset();
}

StreamGuard::~StreamGuard()
{
    if(origin_){
        set_current_stream(origin_.value());
    }
}


} //namespace