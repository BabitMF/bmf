#pragma once

#include "QnnRuntime.h"
#include <vector>

bool deepCopyQnnTensorInfo(Qnn_Tensor_t *dst, const Qnn_Tensor_t *src);

class QnnTensorData {
  public:
    QnnTensorData() = default;
    ~QnnTensorData() = default;
    QnnTensorData(const Qnn_Tensor_t *src);
    QnnTensorData(const Qnn_Tensor_t *src, const Qnn_Tensor_t *next_layer_src);
    QnnTensorData(const QnnTensorData &) = delete;
    QnnTensorData &operator=(const QnnTensorData &) = delete;
    QnnTensorData(QnnTensorData &&) = delete;
    QnnTensorData &operator=(QnnTensorData &&) = delete;
    Qnn_Tensor_t get_tensor() { return *tensor_; };

    bool from_float(float *src);
    bool to_float(float *dst);

    bool from_int(int *src);
    bool to_int(int *dst);

    bool in_place_adjust_quantization();

  private:
    std::vector<uint8_t> owned_data_;
    std::shared_ptr<Qnn_Tensor_t> tensor_ = nullptr;
    std::shared_ptr<Qnn_Tensor_t> tensor_next_layer_ = nullptr;
};
