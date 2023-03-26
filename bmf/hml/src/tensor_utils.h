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

#include <hmp/core/tensor_info.h>


namespace hmp{


struct ShapeStridePair
{
    //Name type instead of a pair/tuple to make sure constuct the vectors inplance
    //and get NRVO
    SizeArray shape;
    SizeArray strides;

    ShapeStridePair() = default;

    ShapeStridePair(const SizeArray &shape_, const SizeArray &strides_)
        : shape(shape_.begin(), shape_.end()),
          strides(strides_.begin(), strides_.end())
    {
    }
};


inline SizeArray inferSize(const SizeArray shape, int64_t nitems)
{
    SizeArray newShape = shape;
    int64_t numAny = 0, anyDim = -1;
    int64_t sizeWithOutAny = 1;
    for(size_t i = 0; i < newShape.size(); ++i){
        if(newShape[i] == -1){
            anyDim = i;
            numAny += 1;
        }
        else{
            sizeWithOutAny *= newShape[i];
        }
    }
    HMP_REQUIRE(numAny <= 1, "Can not determine target shape {}", shape);
    if(numAny == 1){
        HMP_REQUIRE(nitems%sizeWithOutAny == 0, "Can not reshape to {} with nitems={}",
            shape, nitems);
        newShape[anyDim] = nitems / sizeWithOutAny;
    }
    HMP_REQUIRE(nitems == TensorInfo::calcNumel(newShape),
        "Can not reshape to {} with nitems={}", shape, nitems);

    return newShape;
}


// On a high level,
// 1. separate `oldshape` into chunks of dimensions, where the dimensions are
//    ``contiguous'' in each chunk, i.e., oldstride[i] = oldshape[i+1] *
//     oldstride[i+1]
// 2. `newshape` must be able to be separated into same number of chunks as
//    `oldshape` was separated into, where each chunk of newshape has matching
//    ``numel'', i.e., number of subspaces, as the corresponding chunk of
//    `oldshape`.
inline optional<SizeArray> computeStride(const SizeArray &oldShape, 
                        const SizeArray &oldStrides,
                        const SizeArray &newShape)
{
    HMP_REQUIRE(!oldShape.empty() && !oldStrides.empty() && !newShape.empty(), 
        "Invalid argument, empty shape or strdie detected");
    HMP_REQUIRE(oldStrides.size() == oldShape.size(),
        "size of shape and stride are not matched");

    SizeArray newStrides(newShape.size());
    int64_t viewD = newShape.size() - 1;
    //stride for each subspace in the chunk
    int64_t chunkStride = oldStrides.back();
    //nitems in current chunk
    int64_t tensorItems = 1;
    int64_t viewItems = 1;

    for(int64_t tensorD = oldShape.size() - 1; tensorD >= 0; tensorD --){
        tensorItems *= oldShape[tensorD];
        //if end of tensor size or contigous chunk, check the view
        if((tensorD == 0) || 
            (oldShape[tensorD - 1] != 1 && oldStrides[tensorD-1] != tensorItems * chunkStride)){
            while(viewD >= 0 && 
                (viewItems < tensorItems || newShape[viewD] == 1)){
                newStrides[viewD] = viewItems * chunkStride;
                viewItems *= newShape[viewD];
                viewD--;
            }

            if(viewItems != tensorItems){
                return nullopt;
            }
            if(tensorD > 0){
                chunkStride = oldStrides[tensorD - 1];
                tensorItems = 1;
                viewItems = 1;
            }
        }
    }

    if(viewD != -1){
        return nullopt;
    }

    return newStrides;
}

inline ShapeStridePair inferExpandGeometry(
    const SizeArray &tensor_sizes,
    const SizeArray &tensor_strides,
    const SizeArray &sizes)
{
    int64_t ndim = sizes.size();
    int64_t tensor_dim = tensor_sizes.size();

    ShapeStridePair result;
    if (tensor_dim == 0){
        result.shape = sizes;
        result.strides = SizeArray(ndim, 0);
        return result;
    }

    result.shape = SizeArray(ndim);
    result.strides = SizeArray(ndim);

    // create a new geometry for the tensors
    for (int64_t i = ndim - 1; i >= 0; --i){
        int64_t offset = ndim - 1 - i;
        int64_t dim = tensor_dim - 1 - offset;
        int64_t size = (dim >= 0) ? tensor_sizes[dim] : 1;
        int64_t stride = (dim >= 0) ? tensor_strides[dim]
                                    : result.shape[i + 1] * result.strides[i + 1];
        int64_t targetSize = sizes[i];
        if (targetSize == -1){
            HMP_REQUIRE(dim >= 0,
                        "The expanded size of the tensor ({}) isn't allowed in a leading, non-existing dimension {}",
                        targetSize, i)
            targetSize = size;
        }

        if (size != targetSize){
            HMP_REQUIRE(size == 1,
                        "The expanded size of the tensor {} must match the existing size {} at non-singleton dimension {}."
                        " Target sizes: {}, Tensor sizes: {}",
                        targetSize, size, i, sizes, tensor_sizes);

            size = targetSize;
            stride = 0;
        }

        result.shape[i] = size;
        result.strides[i] = stride;
    }

    return result;
}

ShapeStridePair inferSqueezeGeometry(const Tensor &tensor)
{
    ShapeStridePair result;

    for(int64_t d = 0; d < tensor.dim(); ++d){
        if (tensor.size(d) != 1){
            result.shape.push_back(tensor.size(d));
            result.strides.push_back(tensor.stride(d));
        }
    }

    return result;
}

ShapeStridePair inferSqueezeGeometry(const Tensor &tensor, int64_t dim)
{
    ShapeStridePair result;

    for(int64_t d = 0; d < tensor.dim(); ++d){
        if (d != dim || tensor.size(dim) != 1){
            result.shape.push_back(tensor.size(d));
            result.strides.push_back(tensor.stride(d));
        }
    }

    return result;
}


ShapeStridePair inferUnsqueezeGeometry(const Tensor& tensor, int64_t dim)
{
  ShapeStridePair result(tensor.shape(), tensor.strides());
  int64_t new_stride = dim >= tensor.dim() ? 1 : result.shape[dim] * result.strides[dim];
  result.shape.insert(result.shape.begin() + dim, 1);
  result.strides.insert(result.strides.begin() + dim, new_stride);

  return result;
}


void checkSizeArray(const SizeArray &sizes, const char *ctx)
{
    HMP_REQUIRE(sizes.size() > 0, "Empty sizes detected in {}", ctx);
    for(size_t i = 0; i < sizes.size(); ++i){
        HMP_REQUIRE(sizes[i] > 0, 
            "Invalid size {} at dim {} detected in {}", sizes[i], i, ctx);
    }
}

} //namespace