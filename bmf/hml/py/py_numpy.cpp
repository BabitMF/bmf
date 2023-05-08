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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <hmp/tensor.h>
#include <py_type_cast.h>


namespace pybind11{
namespace detail{

const static int NPY_HALF_ = 23;

template <>
struct npy_format_descriptor<hmp::Half> {
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_HALF_);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  static std::string format() {
    // following: https://docs.python.org/3/library/struct.html#format-characters
    return "e";
  }
  static constexpr auto name() {
    return _("float16");
  }
};


}} //



namespace py = pybind11;

using namespace hmp;


static py::dtype scalarTypeToNumpyDtype(const ScalarType scalar_type) 
{
  switch (scalar_type) {
#define TO_FORMAT_STR(scalar_t, scalar_type)\
    case ::hmp::ScalarType::scalar_type: \
        return py::dtype::of<scalar_t>();
    HMP_FORALL_SCALAR_TYPES(TO_FORMAT_STR)
    default:
      throw std::runtime_error(std::string("Got unsupported ScalarType ") + stringfy(scalar_type));
  }
}

static ScalarType numpyDtypeToScalarType(py::dtype dtype) 
{
#define TO_SCALAR_TYPE(scalar_t, scalar_type)\
        if (py::dtype::of<scalar_t>().is(dtype)) {\
            return ::hmp::ScalarType::scalar_type;\
        }
    HMP_FORALL_SCALAR_TYPES(TO_SCALAR_TYPE)

    throw std::runtime_error(std::string("Got unsupported numpy dtype"));
}



Tensor tensor_from_numpy(const py::array& arr)
{
    int ndim = arr.ndim();
    SizeArray shape, strides;
    auto itemsize = arr.itemsize();
    for(int i = 0; i < ndim; ++i){
        auto size = arr.shape()[i];
        auto stride = arr.strides()[i];
        HMP_REQUIRE(stride%itemsize == 0 && stride >= 0,
             "unsupported numpy stride {} at {}", stride, i);

        shape.push_back(static_cast<int64_t>(size));
        strides.push_back(static_cast<int64_t>(stride/itemsize));
    }

    auto buf_info = std::make_shared<pybind11::buffer_info>(arr.request());
    auto ptr = DataPtr(buf_info->ptr, [buf_info](void *ptr) mutable {
                              py::gil_scoped_acquire acquire;
                              //explict release in gil guard
                              buf_info.reset();
                              }, kCPU);

    return from_buffer(
        std::move(ptr),
        numpyDtypeToScalarType(arr.dtype()),
        shape,
        strides);
}

py::array tensor_to_numpy(const Tensor& tensor)
{
    HMP_REQUIRE(tensor.is_cpu(),
         "Only support convert cpu tensor to numpy, got {}", tensor.device_type());

    auto dtype = scalarTypeToNumpyDtype(tensor.scalar_type());
    std::vector<ssize_t> shape, strides;
    auto itemsize = tensor.itemsize();
    for(int i = 0; i < tensor.dim(); ++i){
        shape.push_back(tensor.size(i));
        strides.push_back(tensor.stride(i)*itemsize);
    }

    return py::array(dtype, shape, strides, tensor.unsafe_data(), py::cast(tensor));
}


