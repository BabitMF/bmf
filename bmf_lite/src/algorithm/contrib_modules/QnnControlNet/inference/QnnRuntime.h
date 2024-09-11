#pragma once

#include <QnnInterface.h>
#include <System/QnnSystemInterface.h>
#include <HTP/QnnHtpDevice.h>

#include <dlfcn.h>
#include <iostream>
#include <map>

#include "utils/log.h"

typedef struct QnnFunctionPointers {
    QNN_INTERFACE_VER_TYPE qnnInterface;
    QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;
} QnnFunctionPointers;

using QnnInterface_getProvidersPtr = decltype(&QnnInterface_getProviders);
using QnnSystemInterface_getProvidersPtr =
    decltype(&QnnSystemInterface_getProviders);

class QnnHTPRuntime {
  public:
    QnnHTPRuntime() = default;
    ~QnnHTPRuntime();
    QnnHTPRuntime(const QnnHTPRuntime &) = delete;
    QnnHTPRuntime &operator=(const QnnHTPRuntime &) = delete;
    QnnHTPRuntime(QnnHTPRuntime &&) = delete;
    QnnHTPRuntime &operator=(QnnHTPRuntime &&) = delete;
    bool init(const std::string htp_path, const std::string system_path,
              QnnLog_Level_t log_level = QNN_LOG_LEVEL_INFO,
              QnnLog_Callback_t call_back = (QnnLog_Callback_t) nullptr) {
        if (inited_)
            return true;
        if (!load_libs(htp_path, system_path)) {
            return false;
        }
        if (!init_log(log_level, call_back)) {
            return false;
        }
        if (!init_backend()) {
            return false;
        }
        if (!init_device()) {
            return false;
        }
        if (!setHighPerformanceMode()) {
            return false;
        }
        inited_ = true;
        return true;
    };
    const QnnFunctionPointers &get_qnn_function_ptr() {
        return qnn_function_ptr_;
    }
    const Qnn_DeviceHandle_t get_device_handle() { return device_handle_; }
    const Qnn_BackendHandle_t get_backend_handle() { return backend_handle_; }
    const Qnn_LogHandle_t get_log_handle() { return log_handle_; }

  private:
    bool load_libs(const std::string htp_path, const std::string system_path);
    bool init_log(QnnLog_Level_t log_level = QNN_LOG_LEVEL_INFO,
                  QnnLog_Callback_t call_back = (QnnLog_Callback_t) nullptr);
    bool init_backend();
    bool init_device();
    bool setHighPerformanceMode();
    std::shared_ptr<void> htp_handle_;
    std::shared_ptr<void> system_handle_;
    QnnFunctionPointers qnn_function_ptr_;

    bool inited_ = false;
    bool lib_inited_ = false;
    bool log_inited_ = false;
    bool backend_inited_ = false;
    bool device_inited_ = false;
    bool performance_inited_ = false;
    Qnn_LogHandle_t log_handle_ = nullptr;
    Qnn_BackendHandle_t backend_handle_ = nullptr;
    Qnn_DeviceHandle_t device_handle_ = nullptr;
    QnnHtpDevice_PerfInfrastructure_t *m_perfInfra = nullptr;
    uint32_t m_powerConfigId = 1;
};
