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
#include <iostream>
#include <hmp/core/tensor_info.h>

namespace hmp{


TensorInfo::TensorInfo(const Buffer &buffer, const SizeArray &shape, int64_t bufferOffset)
{
    buffer_ = buffer;
    setSizesAndStrides(shape, bufferOffset);
}

TensorInfo::TensorInfo(const Buffer &buffer, const SizeArray &shape, const SizeArray &strides, int64_t bufferOffset)
{
    buffer_ = buffer;
    setSizesAndStrides(shape, strides, bufferOffset);
}


void TensorInfo::setSizesAndStrides(const SizeArray &shape, int64_t bufferOffset)
{
    auto strides = calcContiguousStrides(shape);
    setSizesAndStrides(shape, strides, bufferOffset);
}

void TensorInfo::setSizesAndStrides(const SizeArray &shape, const SizeArray &strides, int64_t bufferOffset)
{
    HMP_REQUIRE(shape.size() == strides.size(),
         "Invalid size of shape({}) and strides({}) are not matched", shape.size(), strides.size());
    HMP_REQUIRE(bufferOffset >= 0, "Invalid bufferOffset = {}", bufferOffset);
    HMP_REQUIRE(buffer_.defined(), "Buffer is not defined");

    //NOTE: we won't check if shape and strides are out of range
    bufferOffset_ = bufferOffset;
    shape_ = shape;
    strides_ = strides;
    nitems_ = calcNumel(shape);
}


bool TensorInfo::is_contiguous() const
{
    auto cStrides = calcContiguousStrides(shape_);
    for(size_t i = 0; i < cStrides.size(); ++i){
        if(cStrides[i] != strides_[i]){
            return false;
        }
    }
    
    return true;
}

} //namespace