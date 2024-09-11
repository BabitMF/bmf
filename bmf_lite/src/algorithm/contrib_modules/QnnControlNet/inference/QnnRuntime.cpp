#include "QnnRuntime.h"

#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>

bool QnnHTPRuntime::load_libs(const std::string htp_path,
                              const std::string system_path) {
    if (lib_inited_) {
        return true;
    }
    Qnn_ErrorHandle_t error = QNN_SUCCESS;
    auto htp_handle = dlopen(htp_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (htp_handle == nullptr) {
        BMFLITE_LOGE("controlnet", "load error: %s", dlerror());
        return false;
    }
    htp_handle_ = std::shared_ptr<void>(htp_handle, dlclose);
    QnnInterface_getProvidersPtr get_providers =
        reinterpret_cast<QnnInterface_getProvidersPtr>(
            dlsym(htp_handle, "QnnInterface_getProviders"));
    if (get_providers == nullptr) {
        BMFLITE_LOGE("controlnet", "load error");
        return false;
    }
    QnnInterface_t **interfaceProviders{nullptr};
    uint32_t numProviders{0};
    if (get_providers((const QnnInterface_t ***)&interfaceProviders,
                      &numProviders) != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "load error");
        return false;
    }
    if (nullptr == interfaceProviders || numProviders == 0) {
        BMFLITE_LOGE("controlnet", "load error");
        return false;
    }
    bool foundValidInterface{false};
    for (size_t pIdx = 0; pIdx < numProviders; pIdx++) {
        if (QNN_API_VERSION_MAJOR ==
                interfaceProviders[pIdx]->apiVersion.coreApiVersion.major &&
            QNN_API_VERSION_MINOR <=
                interfaceProviders[pIdx]->apiVersion.coreApiVersion.minor) {
            foundValidInterface = true;
            qnn_function_ptr_.qnnInterface =
                interfaceProviders[pIdx]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }

    auto system_handle = dlopen(system_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (system_handle == nullptr) {
        BMFLITE_LOGE("controlnet", "load error");
        return false;
    }
    system_handle_ = std::shared_ptr<void>(system_handle, dlclose);
    QnnSystemInterface_getProvidersPtr system_get_providers =
        reinterpret_cast<QnnSystemInterface_getProvidersPtr>(
            dlsym(system_handle, "QnnSystemInterface_getProviders"));
    if (system_get_providers == nullptr) {
        BMFLITE_LOGE("controlnet", "load error");
        return false;
    }
    QnnSystemInterface_t **systemInterfaceProviders{nullptr};
    uint32_t system_numProviders{0};
    if (system_get_providers(
            (const QnnSystemInterface_t ***)&systemInterfaceProviders,
            &system_numProviders) != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "load error");
        return false;
    }
    if (nullptr == systemInterfaceProviders || system_numProviders == 0) {
        BMFLITE_LOGE("controlnet", "load error");
        return false;
    }
    bool foundValidSystemInterface{false};
    for (size_t pIdx = 0; pIdx < system_numProviders; pIdx++) {
        if (QNN_SYSTEM_API_VERSION_MAJOR ==
                systemInterfaceProviders[pIdx]->systemApiVersion.major &&
            QNN_SYSTEM_API_VERSION_MINOR <=
                systemInterfaceProviders[pIdx]->systemApiVersion.minor) {
            foundValidSystemInterface = true;
            qnn_function_ptr_.qnnSystemInterface =
                systemInterfaceProviders[pIdx]->QNN_SYSTEM_INTERFACE_VER_NAME;
            break;
        }
    }
    if (!foundValidInterface || !foundValidSystemInterface) {
        BMFLITE_LOGE("controlnet", "load error");
        return false;
    }
    lib_inited_ = true;
    return true;
}

bool QnnHTPRuntime::init_log(QnnLog_Level_t logLevel,
                             QnnLog_Callback_t call_back) {
    if (log_inited_ == true) {
        return true;
    }
    if (qnn_function_ptr_.qnnInterface.logCreate == nullptr) {
        BMFLITE_LOGE("controlnet", "log create error");
        return false;
    }
    auto qnnStatus = qnn_function_ptr_.qnnInterface.logCreate(
        call_back, logLevel, &log_handle_);
    if (qnnStatus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "log create error");
        return false;
    }
    log_inited_ = true;
    return true;
}

bool QnnHTPRuntime::init_backend() {
    if (backend_inited_ == true) {
        return true;
    }
    if (qnn_function_ptr_.qnnInterface.backendCreate == nullptr) {
        BMFLITE_LOGE("controlnet", "backend create error");
        return false;
    }
    auto qnnStatus = qnn_function_ptr_.qnnInterface.backendCreate(
        log_handle_, nullptr, &backend_handle_);
    if (qnnStatus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "backend create error");
        return false;
    }
    backend_inited_ = true;
    return true;
}

bool QnnHTPRuntime::init_device() {
    if (device_inited_ == true) {
        return true;
    }
    if (qnn_function_ptr_.qnnInterface.deviceCreate == nullptr) {
        BMFLITE_LOGE("controlnet", "device create error");
        return false;
    }
    auto qnnStatus = qnn_function_ptr_.qnnInterface.deviceCreate(
        log_handle_, nullptr, &device_handle_);
    if (qnnStatus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "device create error");
        return false;
    }
    device_inited_ = true;
    return true;
}

bool QnnHTPRuntime::setHighPerformanceMode() {
    if (performance_inited_ == true) {
        return true;
    }
    if (qnn_function_ptr_.qnnInterface.deviceGetInfrastructure == nullptr) {
        BMFLITE_LOGE("controlnet", "device get infrastructure error");
        return false;
    }
    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    auto qnnStatus =
        qnn_function_ptr_.qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (qnnStatus != QNN_SUCCESS) {
        BMFLITE_LOGE("controlnet", "device get infrastructure error");
        return false;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra =
        static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t *perfInfra = &htpInfra->perfInfra;
    m_perfInfra = perfInfra;
    if (m_perfInfra) {
        uint32_t powerConfigId = 1;
        uint32_t deviceId = 0;
        uint32_t coreId = 0;
        auto qnnStatus =
            perfInfra->createPowerConfigId(deviceId, coreId, &powerConfigId);
        if (qnnStatus != QNN_SUCCESS) {
            BMFLITE_LOGE("controlnet", "setPowerConfig error");
            return false;
        }
        m_powerConfigId = powerConfigId;

        QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
        memset(&powerConfig, 0, sizeof(powerConfig));
        powerConfig.option =
            QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
        powerConfig.dcvsV3Config.dcvsEnable = 0;
        powerConfig.dcvsV3Config.setDcvsEnable = 1;
        powerConfig.dcvsV3Config.contextId = m_powerConfigId;
        powerConfig.dcvsV3Config.powerMode =
            QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
        powerConfig.dcvsV3Config.setSleepLatency = 1;
        powerConfig.dcvsV3Config.setBusParams = 1;
        powerConfig.dcvsV3Config.setCoreParams = 1;
        powerConfig.dcvsV3Config.sleepDisable = 1;
        powerConfig.dcvsV3Config.setSleepDisable = 1;
        uint32_t latencyValue = 40;
        powerConfig.dcvsV3Config.sleepLatency = latencyValue;
        powerConfig.dcvsV3Config.busVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.busVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.busVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.coreVoltageCornerMin =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.coreVoltageCornerTarget =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        powerConfig.dcvsV3Config.coreVoltageCornerMax =
            DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
        const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs[] = {
            &powerConfig, NULL};
        qnnStatus = m_perfInfra->setPowerConfig(m_powerConfigId, powerConfigs);
        if (qnnStatus != QNN_SUCCESS) {
            BMFLITE_LOGE("controlnet", "setPowerConfig error");
            return false;
        }
    } else {
        return false;
    }
    performance_inited_ = true;
    return true;
}
QnnHTPRuntime::~QnnHTPRuntime() {
    if (m_perfInfra) {
        auto qnnStatus = m_perfInfra->destroyPowerConfigId(m_powerConfigId);
        if (qnnStatus != QNN_SUCCESS) {
            BMFLITE_LOGE("controlnet", "destroyPowerConfigId error");
        }
    }
    if (device_handle_) {
        if (qnn_function_ptr_.qnnInterface.deviceFree != nullptr) {
            auto qnnStatus =
                qnn_function_ptr_.qnnInterface.deviceFree(device_handle_);
            if (qnnStatus != QNN_SUCCESS) {
                BMFLITE_LOGE("controlnet", "device free error");
            }
        }
    }
    if (backend_handle_) {
        if (qnn_function_ptr_.qnnInterface.backendFree != nullptr) {
            auto qnnStatus =
                qnn_function_ptr_.qnnInterface.backendFree(backend_handle_);
            if (qnnStatus != QNN_SUCCESS) {
                BMFLITE_LOGE("controlnet", "backend free error");
            }
        }
    }
    if (log_handle_) {
        if (qnn_function_ptr_.qnnInterface.logFree != nullptr) {
            auto qnnStatus =
                qnn_function_ptr_.qnnInterface.logFree(log_handle_);
            if (qnnStatus != QNN_SUCCESS) {
                BMFLITE_LOGE("controlnet", "log free error");
            }
        }
    }
}
