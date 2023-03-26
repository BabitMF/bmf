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

#include <hmp/core/device.h>

namespace hmp{

template<typename F>
class DispatchStub;

template<typename Ret, typename ...Args>
class DispatchStub<Ret(*)(Args...)>
{
public:
    using FuncType = Ret(*)(Args...);

    DispatchStub() = delete;
    DispatchStub(const DispatchStub &) = delete;
    DispatchStub(DispatchStub &&) = delete;

    DispatchStub(const char *name) : name_(name) {}

    Ret operator()(DeviceType device_type, Args ...args)
    {
        auto func = funcs_[static_cast<int>(device_type)];
        HMP_REQUIRE(func != nullptr, 
            "Function {} not implemented in device type {}", name_, device_type);

        return (*func)(std::forward<Args>(args)...);
    }

    template <typename D, typename F>
    friend void setDispatch(D& d, const DeviceType& device, F func);

private:
    void set(DeviceType device, FuncType func)
    {
        funcs_[static_cast<int>(device)] = func;
    }

    const char *name_;
    FuncType funcs_[static_cast<int>(DeviceType::NumDeviceTypes)];
};


#define HMP_DECLARE_DEVICE_DISPATCH(device, name) HMP_DECLARE_TAG(__s##device##name##Dispatch)
#define HMP_DECLARE_ALL_DISPATCH(name) \
        HMP_DECLARE_DEVICE_DISPATCH(kCPU, name); \
        HMP_DECLARE_DEVICE_DISPATCH(kCUDA, name);
#define HMP_IMPORT_DEVICE_DISPATCH(device, name) HMP_IMPORT_TAG(__s##device##name##Dispatch)


#define HMP_DECLARE_DISPATCH_STUB(name, FuncType) \
    using name##Class = DispatchStub<FuncType>; \
    extern name##Class name;        \
    HMP_DECLARE_ALL_DISPATCH(name);

#define HMP_DEFINE_DISPATCH_STUB(name) name##Class name(#name);


template<typename D, typename Func>
void setDispatch(D& d, const DeviceType& device, Func func){
    d.set(device, func);
}


#define HMP_DEVICE_DISPATCH(device, name, func) \
    namespace {\
        static Register<name##Class&, const DeviceType&, decltype(func)> \
            __s##device##name##Dispatch(setDispatch<name##Class&, decltype(func)>, name, device, func); \
    }   \
    HMP_DEFINE_TAG(__s##device##name##Dispatch);



// Scalar type dispatch

#define HMP_TYPE_DISPATCH_CASE(scalar_type, ...) \
    case (ScalarType::scalar_type):{ \
        using scalar_t = getCppType<ScalarType::scalar_type>;\
        return __VA_ARGS__();\
    }


#define HMP_DISPATCH_ALL_TYPES(expectScalarType, name, ...) [&](){\
        switch(expectScalarType){ \
            HMP_TYPE_DISPATCH_CASE(UInt8, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Int8, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(UInt16, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Int16, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Int32, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Int64, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Float32, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Float64, __VA_ARGS__) \
            default: \
                HMP_REQUIRE(false, "{} is not support by {}", expectScalarType, #name); \
        } \
    }()


#define HMP_DISPATCH_ALL_TYPES_AND_HALF(expectScalarType, name, ...) [&](){\
        switch(expectScalarType){ \
            HMP_TYPE_DISPATCH_CASE(UInt8, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Int8, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(UInt16, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Int16, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Int32, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Int64, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Float32, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Float64, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Half, __VA_ARGS__) \
            default: \
                HMP_REQUIRE(false, "{} is not support by {}", expectScalarType, #name); \
        } \
    }()


#define HMP_DISPATCH_UNSIGNED_INTEGRAL_TYPES(expectScalarType, name, ...) [&](){\
        switch(expectScalarType){ \
            HMP_TYPE_DISPATCH_CASE(UInt8, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(UInt16, __VA_ARGS__) \
            default: \
                HMP_REQUIRE(false, "{} is not support by {}", expectScalarType, #name); \
        } \
    }()


#define HMP_DISPATCH_FLOATING_POINT_TYPES(expectScalarType, name, ...) [&](){\
        switch(expectScalarType){ \
            HMP_TYPE_DISPATCH_CASE(Float32, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Float64, __VA_ARGS__) \
            default: \
                HMP_REQUIRE(false, "{} is not support by {}", expectScalarType, #name); \
        } \
    }()


#define HMP_DISPATCH_FLOATING_POINT_TYPES_AND_HALF(expectScalarType, name, ...) [&](){\
        switch(expectScalarType){ \
            HMP_TYPE_DISPATCH_CASE(Half, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Float32, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Float64, __VA_ARGS__) \
            default: \
                HMP_REQUIRE(false, "{} is not support by {}", expectScalarType, #name); \
        } \
    }()


#define HMP_DISPATCH_IMAGE_TYPES_AND_HALF(expectScalarType, name, ...) [&](){\
        switch(expectScalarType){ \
            HMP_TYPE_DISPATCH_CASE(UInt8, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(UInt16, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Float32, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Half, __VA_ARGS__) \
            default: \
                HMP_REQUIRE(false, "{} is not support by {}", expectScalarType, #name); \
        } \
    }()


#define HMP_DISPATCH_FLOAT32_AND_HALF(expectScalarType, name, ...) [&](){\
        switch(expectScalarType){ \
            HMP_TYPE_DISPATCH_CASE(Float32, __VA_ARGS__) \
            HMP_TYPE_DISPATCH_CASE(Half, __VA_ARGS__) \
            default: \
                HMP_REQUIRE(false, "{} is not support by {}", expectScalarType, #name); \
        } \
    }()


} //