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
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/bmf_type_info.h>

namespace bmf_sdk {

class BMF_API BMFAVPacket : public OpaqueDataSet, public SequenceData {
    struct Private;

    std::shared_ptr<Private> self;

  public:
    BMFAVPacket() = default;

    BMFAVPacket(const BMFAVPacket &) = default;

    BMFAVPacket(BMFAVPacket &&) = default;

    BMFAVPacket &operator=(const BMFAVPacket &) = default;

    /**
     * @brief Construct a new BMFAVPacket object
     *
     * @param data contiguous tensor data, cpu only
     */
    BMFAVPacket(const Tensor &data);

    /**
     * @brief Construct a new BMFAVPacket object
     *
     * @param size
     * @param options ref VideoFrame
     */
    BMFAVPacket(int size, const TensorOptions &options = kUInt8);

    /**
     * @brief
     *
     * @tparam Options
     * @param size
     * @param opts ref VideoFrame
     * @return BMFAVPacket
     */
    template <typename... Options>
    static BMFAVPacket make(int size, Options &&...opts) {
        return BMFAVPacket(size, TensorOptions(kUInt8).options(
                                     std::forward<Options>(opts)...));
    }

    /**
     * @brief check if BMFAVPacket if defined
     *
     * @return true
     * @return false
     */
    operator bool() const;

    /**
     * @brief
     *
     * @return Tensor&
     */
    Tensor &data();

    /**
     * @brief
     *
     * @return const Tensor&
     */
    const Tensor &data() const;

    /**
     * @brief return raw pointer of underlying data
     *
     * @return void*
     */
    void *data_ptr();

    /**
     * @brief
     *
     * @return const void*
     */
    const void *data_ptr() const;

    /**
     * @brief number of bytes of underlying data
     *
     * @return int
     */
    int nbytes() const;

    /**
     * @brief copy all extra props(set by member func set_xxx) from
     * `from`(deepcopy if needed),
     *
     * @param from
     * @return VideoFrame&
     */
    BMFAVPacket &copy_props(const BMFAVPacket &from);

    /**
     * @brief get the current data offset which is file write pointer offset
     *
     * @return int64
     */
    int64_t get_offset() const;

    /**
     * @brief get the data whence which is mode. whence == SEEK_SET, from begin;
     * whence == SEEK_CUR, current
     *
     * @return int
     */
    int get_whence() const;

    /**
     * @brief set the current data offset which is file write pointer offset
     *
     * @return void
     */
    void set_offset(int64_t offset);

    /**
     * @brief set the data whence which is mode. whence == SEEK_SET, from begin;
     * whence == SEEK_CUR, current
     *
     * @return void
     */
    void set_whence(int whence);

    int64_t offset_;

    int whence_;
};

} // namespace bmf_sdk

BMF_DEFINE_TYPE(bmf_sdk::BMFAVPacket)
