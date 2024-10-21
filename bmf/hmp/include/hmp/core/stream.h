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
#pragma once

#include <hmp/core/macros.h>
#include <hmp/core/ref_ptr.h>
#include <hmp/core/device.h>

namespace hmp {

using StreamHandle = uint64_t;

struct HMP_API StreamInterface : public RefObject {
    virtual ~StreamInterface() {}

    virtual const Device &device() const = 0;
    virtual StreamHandle handle() const = 0;

    virtual bool query() = 0;
    virtual void synchronize() = 0;
};

class HMP_API Stream {
    RefPtr<StreamInterface> self_;

  public:
    Stream() = delete;
    Stream(const RefPtr<StreamInterface> &self) : self_(self) {}
    Stream(RefPtr<StreamInterface> &&self) : self_(std::move(self)) {}

    bool operator==(const Stream &other) const {
        return self_->handle() == other.handle();
    }

    bool operator!=(const Stream &other) const { return !(*this == other); }

    bool query() { return self_->query(); };
    void synchronize() { self_->synchronize(); };

    const Device &device() const { return self_->device(); }
    StreamHandle handle() const { return self_->handle(); }

    const RefPtr<StreamInterface> &unsafeGet() const { return self_; }
};

HMP_API std::string stringfy(const Stream &stream);

class HMP_API StreamGuard {
    optional<Stream> origin_;

  public:
    StreamGuard() = delete;
    StreamGuard(const StreamGuard &) = delete;
    StreamGuard(StreamGuard &&);

    StreamGuard(const Stream &stream);
    ~StreamGuard();

    optional<Stream> original() { return origin_; }
};

HMP_API Stream create_stream(DeviceType device_type, uint64_t flags = 0);
HMP_API optional<Stream> current_stream(DeviceType device_type);
HMP_API void set_current_stream(const Stream &stream);

namespace impl {

struct HMP_API StreamManager {
    virtual void setCurrent(const Stream &) = 0;
    virtual optional<Stream> getCurrent() const = 0;
    virtual Stream create(uint64_t flags = 0) = 0;
};

HMP_API void registerStreamManager(DeviceType dtype, StreamManager *dm);

#define HMP_REGISTER_STREAM_MANAGER(device, sm)                                \
    namespace {                                                                \
    static Register<DeviceType, ::hmp::impl::StreamManager *>                  \
        __s##device##StreamManager(::hmp::impl::registerStreamManager, device, \
                                   sm);                                        \
    HMP_DEFINE_TAG(__s##device##StreamManager);                                \
    }

#define HMP_DECLARE_STREAM_MANAGER(device)                                     \
    HMP_DECLARE_TAG(__s##device##StreamManager)
#define HMP_IMPORT_STREAM_MANAGER(device)                                      \
    HMP_IMPORT_TAG(__s##device##StreamManager)

} // namespace impl
} // namespace hmp