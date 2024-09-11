#include "QnnTensorData.h"

#include <numeric>
#include <cmath>

template <typename T, typename U>
void quant(T *out, U *in, double scale, double offset, size_t numElements) {
    size_t bitWidth = sizeof(T) * 8;
    double trueBitWidthMax = pow(2, bitWidth) - 1;
    double encodingMin = offset * scale;
    double encodingMax = (trueBitWidthMax + offset) * scale;
    double encodingRange = encodingMax - encodingMin;

    for (size_t i = 0; i < numElements; ++i) {
        int quantizedValue =
            round(trueBitWidthMax * (in[i] - encodingMin) / encodingRange);
        if (quantizedValue < 0) {
            quantizedValue = 0;
        } else if (quantizedValue > (int)trueBitWidthMax) {
            quantizedValue = (int)trueBitWidthMax;
        }
        out[i] = static_cast<T>(quantizedValue);
    }
}

template <typename T, typename U>
void dequant(T *out, U *in, double scale, double offset, size_t numElements) {
    for (size_t i = 0; i < numElements; i++) {
        double quantizedValue = static_cast<double>(in[i]);
        out[i] = static_cast<double>((quantizedValue + offset) * scale);
    }
}

template void quant<uint16_t, float>(uint16_t *out, float *in, double scale,
                                     double offset, size_t numElements);
template void dequant<float, uint16_t>(float *out, uint16_t *in, double scale,
                                       double offset, size_t numElements);

bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src) {
    if (nullptr == dst || nullptr == src) {
        BMFLITE_LOGE("controlnet", "invalid ptr");
        return false;
    }

    dst->version = src->version;
    const char *tensorName = src->v1.name;
    if (!tensorName) {
        dst->v1.name = nullptr;
    } else {
        dst->v1.name = strdup(src->v1.name);
    }
    dst->v1.id = src->v1.id;
    dst->v1.type = src->v1.type;
    dst->v1.dataType = src->v1.dataType;
    dst->v1.quantizeParams.encodingDefinition =
        src->v1.quantizeParams.encodingDefinition;
    dst->v1.quantizeParams.quantizationEncoding =
        src->v1.quantizeParams.quantizationEncoding;
    dst->v1.quantizeParams.scaleOffsetEncoding.scale =
        src->v1.quantizeParams.scaleOffsetEncoding.scale;
    dst->v1.quantizeParams.scaleOffsetEncoding.offset =
        src->v1.quantizeParams.scaleOffsetEncoding.offset;
    dst->v1.rank = src->v1.rank;
    dst->v1.dimensions = (uint32_t *)malloc(sizeof(uint32_t) * src->v1.rank);
    memcpy(dst->v1.dimensions, src->v1.dimensions,
           sizeof(uint32_t) * src->v1.rank);
    size_t size = 0;
    if (src->v1.rank > 0) {
        size = src->v1.dimensions[0];
        for (int i = 1; i < src->v1.rank; i++) {
            size *= src->v1.dimensions[i];
        }
    }
    switch (src->v1.dataType) {
    case QNN_DATATYPE_FLOAT_32:
        size = sizeof(float) * size;
        break;
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_INT_8:
        size = sizeof(uint8_t) * size;
        break;
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
    case QNN_DATATYPE_INT_16:
        size = sizeof(uint16_t) * size;
        break;
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_INT_32:
        size = sizeof(uint32_t) * size;
        break;
    default:
        break;
    }

    dst->v1.memType = QNN_TENSORMEMTYPE_RAW;
    dst->v1.clientBuf.data = nullptr;
    dst->v1.clientBuf.dataSize = size;

    return true;
}

QnnTensorData::QnnTensorData(const Qnn_Tensor_t *src) {
    if (src->version != QNN_TENSOR_VERSION_1) {
        BMFLITE_LOGE("controlnet", "version error");
        return;
    }
    auto deleter = [](Qnn_Tensor_t *p) {
        free((void *)p->v1.name);
        free(p->v1.dimensions);
    };
    tensor_ = std::shared_ptr<Qnn_Tensor_t>(new Qnn_Tensor_t QNN_TENSOR_INIT,
                                            deleter);
    deepCopyQnnTensorInfo(tensor_.get(), src);
    owned_data_.resize(tensor_->v1.clientBuf.dataSize, 0);
    tensor_->v1.clientBuf.data = owned_data_.data();
    return;
}

QnnTensorData::QnnTensorData(const Qnn_Tensor_t *src,
                             const Qnn_Tensor_t *next_layer_src) {
    if (src->version != QNN_TENSOR_VERSION_1 ||
        next_layer_src->version != QNN_TENSOR_VERSION_1) {
        BMFLITE_LOGE("controlnet", "version error");
        return;
    }
    auto deleter = [](Qnn_Tensor_t *p) {
        free((void *)p->v1.name);
        free(p->v1.dimensions);
    };
    tensor_ = std::shared_ptr<Qnn_Tensor_t>(new Qnn_Tensor_t QNN_TENSOR_INIT,
                                            deleter);
    tensor_next_layer_ = std::shared_ptr<Qnn_Tensor_t>(
        new Qnn_Tensor_t QNN_TENSOR_INIT, deleter);
    deepCopyQnnTensorInfo(tensor_.get(), src);
    deepCopyQnnTensorInfo(tensor_next_layer_.get(), next_layer_src);
    owned_data_.resize(tensor_->v1.clientBuf.dataSize, 0);
    tensor_->v1.clientBuf.data = owned_data_.data();
    return;
}

bool QnnTensorData::from_float(float *src) {
    switch (tensor_->v1.dataType) {
    case QNN_DATATYPE_UFIXED_POINT_16:
        quant<uint16_t, float>(
            (uint16_t *)owned_data_.data(), src,
            (double)tensor_->v1.quantizeParams.scaleOffsetEncoding.scale,
            (double)tensor_->v1.quantizeParams.scaleOffsetEncoding.offset,
            owned_data_.size() / 2);
        break;
    default:
        BMFLITE_LOGE("controlnet", "data type error");
        break;
    }
    return true;
}

bool QnnTensorData::to_float(float *dst) {
    switch (tensor_->v1.dataType) {
    case QNN_DATATYPE_UFIXED_POINT_16:
        dequant<float, uint16_t>(
            dst, (uint16_t *)owned_data_.data(),
            (double)tensor_->v1.quantizeParams.scaleOffsetEncoding.scale,
            (double)tensor_->v1.quantizeParams.scaleOffsetEncoding.offset,
            owned_data_.size() / 2);
        break;
    default:
        BMFLITE_LOGE("controlnet", "data type error");
        break;
    }
    return true;
}

bool QnnTensorData::from_int(int *src) {
    switch (tensor_->v1.dataType) {
    case QNN_DATATYPE_INT_32:
        memcpy(owned_data_.data(), src, owned_data_.size() / 4);
        break;
    default:
        BMFLITE_LOGE("controlnet", "data type error");
        break;
    }
    return true;
}

bool QnnTensorData::to_int(int *dst) {
    switch (tensor_->v1.dataType) {
    case QNN_DATATYPE_INT_32:
        memcpy(dst, owned_data_.data(), owned_data_.size() / 4);
        break;
    default:
        BMFLITE_LOGE("controlnet", "data type error");
        break;
    }
    return true;
}

bool QnnTensorData::in_place_adjust_quantization() {
    uint16_t *ptr = (uint16_t *)owned_data_.data();
    auto scale = (double)tensor_->v1.quantizeParams.scaleOffsetEncoding.scale;
    auto offset = (double)tensor_->v1.quantizeParams.scaleOffsetEncoding.offset;
    auto next_layer_scale =
        (double)tensor_next_layer_->v1.quantizeParams.scaleOffsetEncoding.scale;
    auto next_layer_offset =
        (double)
            tensor_next_layer_->v1.quantizeParams.scaleOffsetEncoding.offset;
    double value;
    switch (tensor_->v1.dataType) {
    case QNN_DATATYPE_UFIXED_POINT_16:
        for (int i = 0; i < owned_data_.size() / 2; i++) {
            value = (ptr[i] + offset) * scale;
            value = value / next_layer_scale - next_layer_offset;
            value = value < 0.0 ? 0.0 : value > 65535.0 ? 65535.0 : value;
            ptr[i] = (uint16_t)value;
        }
        break;
    case QNN_DATATYPE_INT_32:
        BMFLITE_LOGE("controlnet", "data type error");
        break;
    }
    return true;
}
