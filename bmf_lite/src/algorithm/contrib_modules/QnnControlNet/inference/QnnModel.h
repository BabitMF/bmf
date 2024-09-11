#pragma once

#include <vector>

#include "QnnRuntime.h"
#include "QnnTensorData.h"

class QnnModel {
  public:
    QnnModel() = default;
    ~QnnModel();
    QnnModel(const QnnModel &) = delete;
    QnnModel &operator=(const QnnModel &) = delete;
    QnnModel(QnnModel &&) = delete;
    QnnModel &operator=(QnnModel &&) = delete;
    virtual bool init(std::shared_ptr<QnnHTPRuntime> runtime,
                      std::string context_binary_path);
    std::vector<std::string> get_all_input_names() {
        std::vector<std::string> tmp_vec;
        for (auto &tensor : input_tensor_vector_) {
            tmp_vec.push_back(std::string(tensor.v1.name));
        }
        return tmp_vec;
    };
    std::vector<std::string> get_all_output_names() {
        std::vector<std::string> tmp_vec;
        for (auto &tensor : output_tensor_vector_) {
            tmp_vec.push_back(std::string(tensor.v1.name));
        }
        return tmp_vec;
    };
    Qnn_Tensor_t *query_input_by_name(std::string name) {
        return input_tensors_[name].get();
    };
    Qnn_Tensor_t *query_output_by_name(std::string name) {
        return output_tensors_[name].get();
    };
    virtual bool register_inout_buffer(std::vector<uint8_t *> &inputs,
                                       std::vector<uint8_t *> &outputs);
    virtual bool register_input_buffer(int index, uint8_t *buffer) {
        if (index > input_tensors_.size()) {
            return false;
        }
        input_tensor_vector_[index].v1.clientBuf.data = buffer;
        return true;
    };
    virtual bool register_output_buffer(int index, uint8_t *buffer) {
        if (index > output_tensors_.size()) {
            return false;
        }
        output_tensor_vector_[index].v1.clientBuf.data = buffer;
        return true;
    };
    virtual bool inference();

  private:
    bool inited_ = false;
    std::shared_ptr<QnnHTPRuntime> htp_runtime_;
    Qnn_ContextHandle_t context_handle_ = nullptr;
    Qnn_GraphHandle_t graph_handle_;
    std::string graph_name_;

    std::map<std::string, std::shared_ptr<Qnn_Tensor_t>> input_tensors_;
    std::map<std::string, std::shared_ptr<Qnn_Tensor_t>> output_tensors_;

    std::vector<Qnn_Tensor_t> input_tensor_vector_;
    std::vector<Qnn_Tensor_t> output_tensor_vector_;

    uint32_t num_input_tensors_;
    uint32_t num_output_tensors_;
};
