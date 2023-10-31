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

#include <memory>
#include <bmf/sdk/hmp_import.h>
#include <bmf/sdk/rational.h>

namespace bmf_sdk {

using OpaqueData = std::shared_ptr<const void>;

/**
 * @brief OpaqueDataInfo for type T, two interface should be implemented
 *  1. const static int OpaqueDataInfo::key = PrivateKey::xxx; //specify the key
 * assicated to type T, key is allocated in PrivateKey
 *  2. static OpaqueData OpaqueDataInfo::cunstruct(...) //make T* manageable
 *
 * @ref ffmpeg_helper.h, test_video_frame.cpp
 *
 * @tparam T
 */
template <typename T> struct OpaqueDataInfo;

/**
 * @brief Keys allocated for private data
 *
 */
struct OpaqueDataKey {
    enum Key {
        // ffmpeg related
        kAVFrame,
        kAVPacket,
        kJsonParam,

        // for future use
        kBMFVideoFrame,
        kATTensor,
        kCVMat,
        kTensor,
        kReserved_7,
        kNumKeys
    };
};

class BMF_API OpaqueDataSet {
  public:
    OpaqueDataSet() = default;

    OpaqueDataSet(OpaqueDataSet &&) = default;

    OpaqueDataSet(const OpaqueDataSet &) = default;

    OpaqueDataSet &operator=(const OpaqueDataSet &) = default;

    /**
     * @brief Attach private data with type T,
     * for type safety, T should be registry by OpaqueDataInfo
     *
     * @ref ffmpeg_helper.h, test_video_frame.cpp
     *
     * @tparam T
     * @tparam Args
     * @param data
     * @param args extra arguments pass to `OpaqueDataInfo<T>::construct(...)`
     */
    template <typename T, typename... Args>
    void private_attach(const T *data, Args &&...args) {
        using Info = OpaqueDataInfo<T>;
        auto opaque = Info::construct(data, std::forward<Args>(args)...);
        set_private_data(Info::key, opaque);
    }

    /**
     * @brief Retrieve readonly private data which attached by private_attach or
     * private_merge
     *
     * @tparam T
     * @return const T*
     */
    template <typename T> const T *private_get() const {
        using Info = OpaqueDataInfo<T>;
        return static_cast<const T *>(private_data(Info::key).get());
    }

    /**
     * @brief merge private data from `from`
     *
     * @param from
     */
    void private_merge(const OpaqueDataSet &from);

    /**
     * @brief utils function to copy props
     *
     * @param from
     * @return OpaqueDataSet&
     */
    OpaqueDataSet &copy_props(const OpaqueDataSet &from);

  protected:
    /**
     * @brief Set the private data object, Derived class can override this
     * function to filter out unsupported keys
     *
     * @param key
     * @param data
     */
    virtual void set_private_data(int key, const OpaqueData &data);
    virtual const OpaqueData &private_data(int key) const;

  private:
    OpaqueData opaque_set_[OpaqueDataKey::kNumKeys] = {nullptr};
}; //

class BMF_API SequenceData {
  public:
    /**
     * @brief Set the pts object
     *
     * @param pts
     */
    void set_pts(int64_t pts) { pts_ = pts; }

    /**
     * @brief
     *
     * @return int
     */
    int64_t pts() const { return pts_; }

    /**
     * @brief Get the time base object
     *
     * @return Rational
     */
    Rational time_base() const { return time_base_; }

    /** @brief set timebase of frame
     *  @param time_base of frame
     */
    void set_time_base(Rational time_base) { time_base_ = time_base; }

    //
    bool operator>(const SequenceData &other) { return pts_ > other.pts_; }

    bool operator>=(const SequenceData &other) { return pts_ >= other.pts_; }

    bool operator<(const SequenceData &other) { return !(*this >= other); }

    bool operator<=(const SequenceData &other) { return !(*this > other); }

    /**
     * @brief util function to copy props
     *
     * @param from
     * @return SequenceData&
     */
    SequenceData &copy_props(const SequenceData &from);

  private:
    // TODO: make all shared
    Rational time_base_;
    int64_t pts_ = 0;
}; //

// support async execution
class BMF_API Future {
    struct Private;

  public:
    Future();
    Future(const Future &) = default;
    Future(Future &&) = default;
    Future &operator=(const Future &) = default;

    virtual ~Future(){};

    /**
     * @brief interface must implemented by sub-class, which provide device info
     *
     * @return Device
     */
    virtual const Device &device() const = 0;

    /**
     * @brief Set the stream object, device specific stream handle
     * currently, only cuda stream handle(cudaStream_t) is suporrted,
     * we only take the ref of this stream, the ownership of this stream
     * is still belongs to caller
     *
     * @param stream
     */
    void set_stream(uint64_t stream);

    /**
     * @brief
     *
     * @return uint64_t
     */
    uint64_t stream() const;

    /**
     * @brief check if result is ready, must be called after record()
     *
     * @return true
     * @return false
     */
    bool ready() const;

    /**
     * @brief record a event to track the readiness of the data
     *
     * @use_current use current stream or self->stream
     *
     */
    void record(bool use_current = true);

    /**
     * @brief force synchronization
     *
     */
    void synchronize();

    /**
     * @brief util function to copy props
     *
     * @param from
     * @return SequenceData&
     */
    Future &copy_props(const Future &from);

  private:
    std::shared_ptr<Private> self;
};

} // namespace bmf_sdk
