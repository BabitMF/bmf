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

#include <bmf/sdk/hmp_import.h>
#include <bmf/sdk/common.h>
#include <bmf/sdk/bmf_type_info.h>
#include <bmf/sdk/timestamp.h>

namespace bmf_sdk {

class Packet;

// like boost::any, but with refcount support
class BMF_API PacketImpl : public RefObject {
    std::function<void(void *)> del_;
    void *obj_ = nullptr;
    const TypeInfo *type_info_ = nullptr;
    int64_t timestamp_ = Timestamp::UNSET;
    double time_ = 0;

  public:
    PacketImpl() = delete;
    PacketImpl(const PacketImpl &) = delete;
    PacketImpl(PacketImpl &&) = default;

    ~PacketImpl();

    template <typename T> T &get() {
        if (bmf_sdk::type_info<T>() != *type_info_) {
            throw std::bad_cast();
        }
        return *static_cast<T *>(obj_);
    }

    template <typename T> const T &get() const {
        if (bmf_sdk::type_info<T>() != *type_info_) {
            throw std::bad_cast();
        }
        return *static_cast<const T *>(obj_);
    }

    template <typename T> bool is() const {
        return bmf_sdk::type_info<T>() == *type_info_;
    }

    const TypeInfo &type_info() const { return *type_info_; }

    void set_timestamp(int64_t timestamp) { timestamp_ = timestamp; }

    int64_t timestamp() const { return timestamp_; }

    void set_time(double time) { time_ = time; }

    double time() const { return time_; }

  protected:
    friend class Packet;
    PacketImpl(void *obj, const TypeInfo *type_info,
               const std::function<void(void *)> &del);
};

class BMF_API Packet {
    RefPtr<PacketImpl> self;

  public:
    Packet() = default;

    template <typename T> Packet(const T &data) : Packet(new T(data)) {}

    Packet(const Packet &data) : self(data.self) {}

    template <typename T> Packet(T &data) : Packet(new T(data)) {}

    Packet(Packet &data) : self(data.self) {}

    template <typename T> Packet(T &&data) : Packet(new T(std::move(data))) {}

    Packet(Packet &&data) : self(std::move(data.self)) {}

    Packet(RefPtr<PacketImpl> &impl) : self(impl) {}

    Packet &operator=(const Packet &other) = default;

    operator bool() const { return bool(self); }

    template <typename T> T &get() { return self->get<T>(); }

    template <typename T> const T &get() const { return self->get<T>(); }

    template <typename T> bool is() const { return self->is<T>(); }

    const TypeInfo &type_info() const;

    //
    void set_timestamp(int64_t timestamp);
    int64_t timestamp() const;

    //
    void set_time(double time);
    double time() const;

    PacketImpl *unsafe_self();
    const PacketImpl *unsafe_self() const;

    //
    static Packet generate_eos_packet();

    static Packet generate_eof_packet();

  protected:
    template <typename T> Packet(T *obj) {
        auto impl = new PacketImpl(obj, &bmf_sdk::type_info<T>(),
                                   [](void *obj) { delete (T *)obj; });
        self = RefPtr<PacketImpl>::take(impl, true);
    }
};

} // namespace bmf_sdk

BMF_DEFINE_TYPE(std::string);
