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


#include <hmp/tensor.h>
#ifdef HMP_ENABLE_TORCH
#include <hmp/torch/torch.h>
//cast implementation
#include <torch/csrc/utils/pybind.h>
#endif

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <py_type_cast.h>
#include <py_utils.h>

namespace py = pybind11;

using namespace hmp;

Tensor tensor_from_numpy(const py::array& arr);
py::array tensor_to_numpy(const Tensor& tensor);

Device parse_device(const py::object &obj, const Device &ref)
{
    Device device(ref);
    if (PyUnicode_Check(obj.ptr())){
        device = Device(py::cast<std::string>(obj));
    }
    else if (py::isinstance<Device>(obj)){
        device = py::cast<Device>(obj);
    }
    else{
        try{
            device = py::cast<DeviceType>(obj);
        }
        catch(std::bad_cast &){
        }
    }

    return device;
}

TensorOptions parse_tensor_options(const py::kwargs &kwargs, const TensorOptions &ref)
{
    TensorOptions opts(ref);

    if (kwargs.contains("pinned_memory")){
      opts = opts.pinned_memory(py::cast<bool>(kwargs["pinned_memory"]));
    }

    if (kwargs.contains("device")){
        opts = opts.device(parse_device(kwargs["device"]));
    }

    if (kwargs.contains("dtype")){
      opts = opts.dtype(py::cast<ScalarType>(kwargs["dtype"]));
    }

    return opts;
}

void tensorBind(py::module &m)
{
    //
    m.def("from_numpy", [](const py::array& arr){
      return tensor_from_numpy(arr);
    })
    .def("from_numpy", [](const py::list& arr_list){
      py::list tensor_list;
      for(auto &arr : arr_list){
        tensor_list.append(py::cast(tensor_from_numpy(py::cast<py::array>(arr))));
      }
        return tensor_list;
    })
    .def("to_numpy", [](const Tensor &tensor){
      return tensor_to_numpy(tensor);
    })
    .def("to_numpy", [](const TensorList &tensors){
      py::list arr_list;
      for(auto &tensor : tensors){
        arr_list.append(tensor_to_numpy(tensor));
      }
      return arr_list;
    })
#ifdef HMP_ENABLE_TORCH
    .def("from_torch", [](const at::Tensor &t){
      return hmp::torch::from_tensor(t);
    })
#endif
    .def("empty", [](const SizeArray &shape, const py::kwargs &kwargs){
        auto options = parse_tensor_options(kwargs);
        return empty(shape, options);
    })
    .def("empty_like", [](const Tensor &other, const py::kwargs &kwargs){
        auto options = parse_tensor_options(kwargs, other.options());
        return empty_like(other, options);
    })
    .def("zeros", [](const SizeArray &shape, const py::kwargs &kwargs){
        auto options = parse_tensor_options(kwargs);
        return zeros(shape, options);
    })
    .def("zeros_like", [](const Tensor &other, const py::kwargs &kwargs){
        auto options = parse_tensor_options(kwargs, other.options());
        return zeros_like(other, options);
    })
    .def("ones", [](const SizeArray &shape, const py::kwargs &kwargs){
        auto options = parse_tensor_options(kwargs);
        return ones(shape, options);
    })
    .def("ones_like", [](const Tensor &other, const py::kwargs &kwargs){
        auto options = parse_tensor_options(kwargs, other.options());
        return ones_like(other, options);
    })
    .def("arange", [](int64_t start, int64_t end, int64_t step, const py::kwargs &kwargs){
        auto options = parse_tensor_options(kwargs);
        return arange(start, end, step, options);
    }, py::arg("start"), py::arg("end"), py::arg("step")=1)
    .def("arange", [](int64_t end, const py::kwargs &kwargs){
        auto options = parse_tensor_options(kwargs);
        return arange(0, end, 1, options);
    }, py::arg("end"))
    .def("copy", &copy)

    //shape transformation
    .def("concat", (Tensor(*)(const TensorList&, int64_t))&concat,
                    py::arg("tensors"), py::arg("axis") = 0)
    .def("concat", (Tensor&(*)(Tensor&, const TensorList&, int64_t))&concat,
                    py::arg("out"), py::arg("tensors"), py::arg("axis") = 0)
    .def("stack", (Tensor(*)(const TensorList&, int64_t))&stack,
                    py::arg("tensors"), py::arg("axis") = 0)
    .def("stack", (Tensor&(*)(Tensor&, const TensorList&, int64_t))&stack,
                    py::arg("out"), py::arg("tensors"), py::arg("axis") = 0)
    .def("vstack", (Tensor(*)(const TensorList&))&vstack, py::arg("tensors"))
    .def("vstack", (Tensor&(*)(Tensor&, const TensorList&))&vstack,
                    py::arg("out"), py::arg("tensors"))
    .def("hstack", (Tensor(*)(const TensorList&))&hstack, py::arg("tensors"))
    .def("hstack", (Tensor&(*)(Tensor&, const TensorList&))&hstack,
                    py::arg("out"), py::arg("tensors"))
    
    //file io
    .def("fromfile", &fromfile, py::arg("fn"), py::arg("dtype"), py::arg("count")=-1,
                     py::arg("offset")=0)
    .def("tofile", &tofile, py::arg("data"), py::arg("fn"))
    ;

    //
    py::class_<Tensor>(m, "Tensor")
        .def("__str__", [](const Tensor &self){
            return stringfy(self);
        })
        .def("__repr__", [](const Tensor &self){
            return self.repr();
        })
        .def_property_readonly("defined", &Tensor::defined)
        .def_property_readonly("device", &Tensor::device)
        .def_property_readonly("device_type", &Tensor::device_type)
        .def_property_readonly("device_index", &Tensor::device_index)
        .def_property_readonly("dtype", &Tensor::dtype)
        .def_property_readonly("shape", [](const Tensor &self){
          py::tuple shape = py::cast(self.shape());
          return shape;
        })
        .def_property_readonly("strides", [](const Tensor &self){
            py::tuple shape = py::cast(self.strides());
            return shape;
        })
        .def_property_readonly("dim", &Tensor::dim)
        .def("size", &Tensor::size)
        .def("stride", &Tensor::stride)
        .def_property_readonly("nbytes", &Tensor::nbytes)
        .def_property_readonly("itemsize", &Tensor::itemsize)
        .def_property_readonly("nitems", &Tensor::nitems)
        .def_property_readonly("is_contiguous", &Tensor::is_contiguous)
        .def_property_readonly("is_cpu", &Tensor::is_cpu)
        .def_property_readonly("is_cuda", &Tensor::is_cuda)
        .def("fill_", &Tensor::fill_)

        //
        .def("alias", &Tensor::alias)
        .def("view", &Tensor::view)
        .def("clone", &Tensor::clone)
        .def("as_strided", &Tensor::as_strided, py::arg("shape"), py::arg("strides"), py::arg("offset")=py::none())
        .def("as_strided_", &Tensor::as_strided_, py::arg("shape"), py::arg("strides"), py::arg("offset")=py::none())
        .def("squeeze", &Tensor::squeeze, py::arg("dim")=py::none())
        .def("squeeze_", &Tensor::squeeze_, py::arg("dim")=py::none())
        .def("unsqueeze", &Tensor::unsqueeze, py::arg("dim")=0)
        .def("unsqueeze_", &Tensor::unsqueeze_, py::arg("dim")=0)

        //
        .def("reshape", &Tensor::reshape)
        .def("transpose", &Tensor::transpose)
        .def("permute", &Tensor::permute)
        .def("slice", &Tensor::slice, py::arg("dim"), py::arg("start"), py::arg("end")=py::none(), py::arg("step")=1)
        .def("select", &Tensor::select)

        .def("to", (Tensor(Tensor::*)(const Device&, bool) const)&Tensor::to, py::arg("device"), py::arg("non_blocking")=false)
        .def("to", (Tensor(Tensor::*)(DeviceType, bool) const)&Tensor::to, py::arg("device"), py::arg("non_blocking")=false)
        .def("to", [](const Tensor &self, const std::string &deviceStr){
            return self.to(Device(deviceStr));
        }, py::arg("device"))
        .def("to", (Tensor(Tensor::*)(ScalarType) const)&Tensor::to, py::arg("dtype"))
        .def("copy_", &Tensor::copy_, py::arg("src"))

        .def("contiguous", &Tensor::contiguous)
        .def("cpu", &Tensor::cpu, py::arg("non_blocking")=false)
        .def("cuda", &Tensor::cuda)
        .def("numpy", [](const Tensor &self){
          return tensor_to_numpy(self);
        })
#ifdef HMP_ENABLE_TORCH
        .def("torch", [](const Tensor &self){
          return hmp::torch::tensor(self);
        })
#endif

        // Unary ops
        .def("round", &Tensor::round)
        .def("round_", &Tensor::round_)
        .def("ceil", &Tensor::ceil)
        .def("ceil_", &Tensor::ceil_)
        .def("floor", &Tensor::floor)
        .def("floor_", &Tensor::floor_)
        .def("abs", &Tensor::abs)
        .def("abs_", &Tensor::abs_)
        .def("clip", &Tensor::clip, py::arg("min"), py::arg("max"))
        .def("clip_", &Tensor::clip_, py::arg("min"), py::arg("max"))

        // Binary ops
        #define BIND_TENSOR_BOP(name, op) \
          .def(py::self op py::self) \
          .def(py::self op##= py::self) \
          .def(py::self op Scalar()) \
          .def(py::self op##= Scalar()) \
          .def(Scalar() op py::self)  \
          .def(#name, (Tensor(Tensor::*)(const Tensor &b) const)&Tensor::name)  \
          .def(#name, (Tensor(Tensor::*)(const Scalar &b) const)&Tensor::name)  \
          .def(#name "_", (Tensor&(Tensor::*)(const Tensor &b))&Tensor::name##_)  \
          .def(#name "_", (Tensor&(Tensor::*)(const Scalar &b))&Tensor::name##_) 

        BIND_TENSOR_BOP(add, +)
        BIND_TENSOR_BOP(sub, -)
        BIND_TENSOR_BOP(mul, *)
        BIND_TENSOR_BOP(div, /)

        //shape transformation
        .def("flatten", &Tensor::flatten)

        .def("tofile", &Tensor::tofile, py::arg("fn"))
      ;
}