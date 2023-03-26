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

#include <kernel/tensor_factory.h>
#include <hmp/format.h>

namespace hmp{
namespace kernel{

HMP_DEFINE_DISPATCH_STUB(fill_stub)
HMP_DEFINE_DISPATCH_STUB(copy_stub)
HMP_DEFINE_DISPATCH_STUB(arange_stub)

Tensor empty(const SizeArray &shape, const TensorOptions &options)
{
    int64_t nitems = TensorInfo::calcNumel(shape);
    unsigned flags = 0;
    if(options.pinned_memory()){
        flags |= static_cast<unsigned>(AllocatorFlags::Pinned);
    }

    auto scalar_type = options.scalar_type();
    auto device_type = options.device().type();
    auto allocator = get_allocator(device_type, flags);
    HMP_REQUIRE(allocator, "Device type {} is not supported", device_type);
    HMP_REQUIRE(nitems > 0, "Invalid tensor shape={}", shape);

    return Tensor(makeRefPtr<TensorInfo>(
        Buffer(scalar_type, nitems, allocator, options.pinned_memory()), shape));
}


Tensor& copy(Tensor &self, const Tensor &other)
{
#ifndef HMP_BUILD_SHARED
    HMP_IMPORT_DEVICE_DISPATCH(kCPU, copy_stub);
#ifdef HMP_EANBLE_CUDA
    HMP_IMPORT_DEVICE_DISPATCH(kCUDA, copy_stub);
#endif
#endif

    HMP_REQUIRE(self.shape() == other.shape(),
         "copy: can not copy data from shape {}, expect shape {}", other.shape(), self.shape());

    //always do copy in device kernel
    auto device_type = self.device_type() == kCPU ? other.device_type() : self.device_type();
    return copy_stub(device_type, self, other);
}


}} //