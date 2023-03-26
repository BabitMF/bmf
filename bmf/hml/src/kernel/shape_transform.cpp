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
#include <kernel/shape_transform.h>
#include <kernel/kernel_utils.h>

namespace hmp{
namespace kernel{


static SizeArray calcConcatShape(const TensorList &tensors, int64_t axis)
{
    SizeArray shape(tensors[0].shape());
    HMP_REQUIRE(axis < shape.size(), 
        "concat: axis({}) is out of range({})", axis, shape.size());

    for(int64_t i = 1; i < tensors.size(); ++i){
        HMP_REQUIRE(tensors[i].dim() == shape.size(), 
        "concat: {}th tensor has invalid dim({}), expect {}",
        i, tensors[i].dim(), shape.size());

        shape[axis] += tensors[i].size(axis);
    }
    return shape;
}


static SizeArray calcStackShape(const TensorList &tensors, int64_t axis)
{
    SizeArray shape(tensors[0].shape());
    HMP_REQUIRE(axis <= shape.size(), 
        "stack: axis({}) is out of range({})", axis, shape.size() + 1);
    shape.insert(shape.begin() + axis, 1);

    for(int64_t i = 1; i < tensors.size(); ++i){
        HMP_REQUIRE(tensors[i].shape() == tensors[0].shape(), 
            "stack: {}th tensor has invalid shape({}), expect {}",
             i, tensors[i].shape(), tensors[0].shape());
        shape[axis] += 1;
    }

    return shape;
}


Tensor& concat(Tensor &out, const TensorList &tensors, int64_t axis)
{
    axis = wrap_size(axis, tensors[0].dim());
    auto shape = calcConcatShape(tensors, axis);

    HMP_REQUIRE(out.shape() == shape, "concat: expect out has shape {}, got {}",
        shape, out.shape());

    for(int64_t i = 0, si = 0; i < tensors.size(); ++i){
        auto &t = tensors[i];
        auto tmp = out.slice(axis, si, si + t.size(axis));
        copy(tmp, t);
        si += t.size(axis);
    }

    return out;
}


Tensor concat(const TensorList &tensors, int64_t axis)
{
    axis = wrap_size(axis, tensors[0].dim());
    auto shape = calcConcatShape(tensors, axis);
    auto out = empty(shape, tensors[0].options());

    kernel::concat(out, tensors, axis);

    return out;
}


Tensor& stack(Tensor &out, const TensorList &tensors, int64_t axis)
{
    axis = wrap_size(axis, tensors[0].dim() + 1);
    auto shape = calcStackShape(tensors, axis);

    HMP_REQUIRE(out.shape() == shape, "stack: expect out has shape {}, got {}",
        shape, out.shape());

    for(int64_t i = 0; i < tensors.size(); ++i){
        auto &t = tensors[i];
        auto tmp = out.select(axis, i);
        copy(tmp, t);
    }

    return out;
}

Tensor stack(const TensorList &tensors, int64_t axis)
{
    axis = wrap_size(axis, tensors[0].dim() + 1);
    auto shape = calcStackShape(tensors, axis);
    auto out = empty(shape, tensors[0].options());

    kernel::stack(out, tensors, axis);

    return out;
}


Tensor atleast_2d(const Tensor &in)
{
    if(in.dim() < 2){
        return in.reshape({1, -1});
    }
    else{
        return in;
    }
}


Tensor& vstack(Tensor &out, const TensorList &tensors_)
{
    TensorList tensors;
    for(auto &t : tensors_){
        tensors.push_back(atleast_2d(t));
    }

    return kernel::concat(out, tensors, 0);
}

Tensor vstack(const TensorList &tensors_)
{
    TensorList tensors;
    for(auto &t : tensors_){
        tensors.push_back(atleast_2d(t));
    }

    return kernel::concat(tensors, 0);
}

Tensor& hstack(Tensor &out, const TensorList &tensors)
{
    return kernel::concat(out, tensors, -1);
}

Tensor hstack(const TensorList &tensors)
{
    return kernel::concat(tensors, -1);
}

}} //namespace