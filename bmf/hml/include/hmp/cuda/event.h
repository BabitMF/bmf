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
#include <hmp/cuda/macros.h>
#include <hmp/core/stream.h>

namespace hmp {
namespace cuda {

class HMP_API Event {
    void *event_;
    unsigned flags_;
    bool is_created_;
    int device_index_;

  public:
    Event(const Event &) = delete;
    Event();
    Event(Event &&other);
    Event(bool enable_timing, bool blocking = true, bool interprocess = false);
    ~Event();

    bool is_created() const { return is_created_; }

    void record(const optional<Stream> &stream = nullopt);
    void block(const optional<Stream> &stream = nullopt);
    bool query() const;
    void synchronize() const;

    float elapsed(const Event &other);
};
} // namespace cuda
}; // namespace hmp