#include "QnnModel.h"

#include <fstream>
#include <numeric>
#include <cmath>

bool QnnModel::init(std::shared_ptr<QnnHTPRuntime> runtime,
                    std::string context_binary_path) {
    if (inited_ == true) {
        return true;
    }
    htp_runtime_ = runtime;
    std::ifstream context_binary(context_binary_path, std::ios::binary);
    if (!context_binary.is_open()) {
        BMFLITE_LOGE("controlnet", "open error");
        return false;
    }
    context_binary.seekg(0, std::ios::end);
    size_t size = context_binary.tellg();
    context_binary.seekg(0, std::ios::beg);
    std::vector<uint8_t> context_binary_data(size);
    context_binary.read((char *)context_binary_data.data(), size);
    context_binary.close();
    auto qnn_function_ptr = htp_runtime_->get_qnn_function_ptr();
    auto log_handle = htp_runtime_->get_log_handle();
    auto backend_handle = htp_runtime_->get_backend_handle();
    auto device_handle = htp_runtime_->get_device_handle();

    QnnSystemContext_Handle_t sysCtxHandle{nullptr};
    if (qnn_function_ptr.qnnSystemInterface.systemContextCreate == nullptr ||
        qnn_function_ptr.qnnSystemInterface.systemContextFree == nullptr ||
        qnn_function_ptr.qnnSystemInterface.systemContextGetBinaryInfo ==
            nullptr) {
        BMFLITE_LOGE(
            "controlnet",
            "qnnSystemInterface.systemContextCreateFromBinary is null");
        return false;
    }
    auto qnnStatus =
        qnn_function_ptr.qnnSystemInterface.systemContextCreate(&sysCtxHandle);
    if (qnnStatus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "create system context error");
        return false;
    }
    const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};
    qnnStatus = qnn_function_ptr.qnnSystemInterface.systemContextGetBinaryInfo(
        sysCtxHandle, context_binary_data.data(), size, &binaryInfo,
        &binaryInfoSize);
    if (qnnStatus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "get system context binary info error");
        qnn_function_ptr.qnnSystemInterface.systemContextFree(sysCtxHandle);
        return false;
    }
    QnnSystemContext_GraphInfo_t *graphs =
        binaryInfo->contextBinaryInfoV1.graphs;
    QnnSystemContext_GraphInfoV1_t graphInfo = graphs[0].graphInfoV1;
    graph_name_ = std::string(graphInfo.graphName);
    num_input_tensors_ = graphInfo.numGraphInputs;
    num_output_tensors_ = graphInfo.numGraphOutputs;
    auto deleter = [](Qnn_Tensor_t *p) {
        free((void *)p->v1.name);
        free(p->v1.dimensions);
    };
    for (int i = 0; i < num_input_tensors_; i++) {
        auto ptr = std::shared_ptr<Qnn_Tensor_t>(
            new Qnn_Tensor_t QNN_TENSOR_INIT, deleter);
        deepCopyQnnTensorInfo(ptr.get(), &graphInfo.graphInputs[i]);
        input_tensors_[std::string(graphInfo.graphInputs[i].v1.name)] = ptr;
        input_tensor_vector_.push_back(*ptr);
    }

    for (int i = 0; i < num_output_tensors_; i++) {
        auto ptr = std::shared_ptr<Qnn_Tensor_t>(
            new Qnn_Tensor_t QNN_TENSOR_INIT, deleter);
        deepCopyQnnTensorInfo(ptr.get(), &graphInfo.graphOutputs[i]);
        output_tensors_[std::string(graphInfo.graphOutputs[i].v1.name)] = ptr;
        output_tensor_vector_.push_back(*ptr);
    }

    qnn_function_ptr.qnnSystemInterface.systemContextFree(sysCtxHandle);

    qnnStatus = qnn_function_ptr.qnnInterface.contextCreateFromBinary(
        backend_handle, device_handle, nullptr, context_binary_data.data(),
        size, &context_handle_, nullptr);
    if (qnnStatus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "create context from binary error");
        return false;
    }

    qnnStatus = qnn_function_ptr.qnnInterface.graphRetrieve(
        context_handle_, graph_name_.c_str(), &graph_handle_);
    if (qnnStatus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "retrieve graph error");
        return false;
    }

    inited_ = true;
    return true;
}

QnnModel::~QnnModel() {
    auto qnn_function_ptr = htp_runtime_->get_qnn_function_ptr();
    if (context_handle_ != nullptr) {
        qnn_function_ptr.qnnInterface.contextFree(context_handle_, nullptr);
    }
}

bool QnnModel::register_inout_buffer(std::vector<uint8_t *> &inputs,
                                     std::vector<uint8_t *> &outputs) {
    if (!inited_) {
        BMFLITE_LOGE("controlnet", "model not inited");
        return false;
    }

    Qnn_Tensor_t *ptr = input_tensor_vector_.data();
    for (int i = 0; i < num_input_tensors_; i++) {
        if (inputs[i] == nullptr) {
            BMFLITE_LOGE("controlnet", "input buffer is null");
            return false;
        }
        ptr[i].v1.clientBuf.data = (void *)inputs[i];
    }
    ptr = output_tensor_vector_.data();
    for (int i = 0; i < num_output_tensors_; i++) {
        if (outputs[i] == nullptr) {
            BMFLITE_LOGE("controlnet", "output buffer is null");
            return false;
        }
        ptr[i].v1.clientBuf.data = (void *)outputs[i];
    }

    return true;
}

bool QnnModel::inference() {
    if (!inited_) {
        BMFLITE_LOGE("controlnet", "model not inited");
        return false;
    }
    auto qnn_function_ptr = htp_runtime_->get_qnn_function_ptr();
    auto qnn_satus = qnn_function_ptr.qnnInterface.graphExecute(
        graph_handle_, input_tensor_vector_.data(), num_input_tensors_,
        output_tensor_vector_.data(), num_output_tensors_, nullptr, nullptr);
    if (qnn_satus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "graph execute error %lu \n", qnn_satus);
        return false;
    }
    return true;
}