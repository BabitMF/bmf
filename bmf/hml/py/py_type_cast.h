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
#include <pybind11/pybind11.h>
#include <hmp/tensor.h>


namespace pybind11 { namespace detail {

template<> struct type_caster<hmp::Scalar> {
    PYBIND11_TYPE_CASTER(hmp::Scalar, _("Scalar"));

    bool load(handle src, bool)
    {
        auto obj = src.ptr();
        if(PyFloat_Check(obj)){
            value = src.cast<double>();
        }
        else if(PyLong_Check(obj)){
            value = src.cast<int64_t>();
        }
        else if(PyBool_Check(obj)){
            value = src.cast<bool>();
        }
        else{
            throw std::runtime_error("Unsupported scalar type");
        }

        return true;
    }

    static handle cast(hmp::Scalar src, return_value_policy, handle){
        if(src.is_integral(false)){
            return PyLong_FromLongLong(src.to<int64_t>());
        }
        else if(src.is_boolean()){
            return PyBool_FromLong(src.to<bool>());
        }
        else if(src.is_floating_point()){
            return PyFloat_FromDouble(src.to<double>());
        }
        else{
            throw std::runtime_error("unexpected Scalar type");
        }
    }
};


template<typename T> struct type_caster<hmp::optional<T>> {
    PYBIND11_TYPE_CASTER(hmp::optional<T>, _("hmp::optional"));

    bool load(handle src, bool b)
    {
        if (src.is_none()) {
            value = hmp::nullopt;
        }
        else {
            value = pybind11::cast<T>(src);
        }

        return true;
    }

    static handle cast(hmp::optional<T> src, return_value_policy policy, handle hnd){
        if (src.has_value()) {
            return type_caster<T>::cast(*src, policy, hnd);
        }
        else {
            return none();
        }
    }
};


}}; //namespace pybind11::detail
