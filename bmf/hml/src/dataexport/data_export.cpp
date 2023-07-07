#include <hmp/dataexport/data_export.h>

#include <iostream>

namespace hmp{
static DLDataType getDLDataType(const Tensor& t) {
    DLDataType dtype;
    dtype.lanes = 1;
    dtype.bits = sizeof_scalar_type(t.scalar_type()) * 8;
    switch (t.scalar_type()) {
        case ScalarType::UInt8:
        case ScalarType::UInt16:
        // case ScalarType::UInt32:
        // case ScalarType::UInt64:
        dtype.code = DLDataTypeCode::kDLUInt;
        break;
        case ScalarType::Int8:
        case ScalarType::Int16:
        case ScalarType::Int32:
        case ScalarType::Int64:
        dtype.code = DLDataTypeCode::kDLInt;
        break;
        case ScalarType::Float64:
        case ScalarType::Float32:
        case ScalarType::Half:
        dtype.code = DLDataTypeCode::kDLFloat;
        break;
        case ScalarType::Undefined:
        default:
        HMP_REQUIRE(false, "Undefined is not a valid ScalarType");
    }
    return dtype;
}

static DLDevice getDLDevice(const Tensor& tensor, const int64_t& device_id) {
    DLDevice ctx;
    ctx.device_id = device_id;
    switch (tensor.device().type()) {
        case DeviceType::CPU:
        ctx.device_type = DLDeviceType::kDLCPU;
        break;
        case DeviceType::CUDA:
        ctx.device_type = DLDeviceType::kDLCUDA;
        break;
        default:
        HMP_REQUIRE(false, "Cannot pack tensors on " + stringfy(tensor.device()));
    }
    return ctx;
}

struct HmpDLMTensor {
    // HmpDLMTensor() { std::cout << "Construct HmpDLMTensor\n"; }
    Tensor handle;
    DLManagedTensor tensor;
};

void deleter(DLManagedTensor* arg) {
    // std::cout << "Destruct HmpDLMTensor\n";
    delete static_cast<HmpDLMTensor*>(arg->manager_ctx);
}

DLManagedTensor* to_dlpack(const Tensor& src) {

    HmpDLMTensor* hmpDLMTensor(new HmpDLMTensor);
    hmpDLMTensor->handle = src;
    hmpDLMTensor->tensor.manager_ctx = hmpDLMTensor;
    hmpDLMTensor->tensor.deleter = &deleter;
    hmpDLMTensor->tensor.dl_tensor.data = src.unsafe_data();
    int64_t device_id = 0;
    if (src.is_cuda()) {
        device_id = src.device_index();
    }
    hmpDLMTensor->tensor.dl_tensor.device = getDLDevice(src, device_id);
    hmpDLMTensor->tensor.dl_tensor.ndim = src.dim();
    hmpDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src);
    hmpDLMTensor->tensor.dl_tensor.shape =
        const_cast<int64_t*>(src.shape().data());
    hmpDLMTensor->tensor.dl_tensor.strides =
        const_cast<int64_t*>(src.strides().data());
    hmpDLMTensor->tensor.dl_tensor.byte_offset = 0;
    return &(hmpDLMTensor->tensor);
}

} // namespace hmp