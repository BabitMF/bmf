/*
 * Copyright 2024 Babit Authors
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

#include "common/error_code.h"
#include "media/video_buffer/hardware_device_context.h"
#include "media/video_buffer/gl_texture_video_buffer/egl_hardware_device_context.h"
#include "media/video_buffer/metal_texture_video_buffer/mtl_device_context.h"

namespace bmf_lite {

int HWDeviceContextManager::getCurrentHwDeviceContext(
    const HWDeviceType device_type, std::shared_ptr<HWDeviceContext> &context) {
    if (device_type == kHWDeviceTypeEGLCtx) {
#ifdef BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER
        std::shared_ptr<EglHWDeviceContext> egl_context =
            std::make_shared<EglHWDeviceContext>();
        context = egl_context->storeCurrent();
        return BMF_LITE_StsOk;
#endif
    }
    if (device_type == kHWDeviceTypeMTL) {
#ifdef BMF_LITE_ENABLE_METALBUFFER
        std::shared_ptr<MtlDeviceContext> egl_context =
            std::make_shared<MtlDeviceContext>();
        context = egl_context->storeCurrent();
        return BMF_LITE_StsOk;
#endif
    }
    if (device_type == kHWDeviceTypeNone) {
#ifdef BMF_LITE_ENABLE_CPUMEMORYBUFFER
        return BMF_LITE_StsOk;
#endif
    }

    return BMF_LITE_StsBadArg;
}

int HWDeviceContextManager::createHwDeviceContext(
    HardwareDeviceCreateInfo *create_info,
    std::shared_ptr<HWDeviceContext> &context) {
    if (create_info == NULL) {
        // BMF_LITE_ERR("create info is NULL");
        return BMF_LITE_StsBadArg;
    }

    if (create_info->device_type == kHWDeviceTypeEGLCtx) {
#ifdef BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER
        EGLContextInfo *egl_info =
            (EGLContextInfo *)(create_info->context_info);

        std::shared_ptr<EglHWDeviceContext> egl_context =
            std::make_shared<EglHWDeviceContext>();
        int res = egl_context->create_context(egl_info->egl_context);
        if (res < 0) {
            return res;
        }
        context = egl_context;
        return 0;
#endif
    }
    if (create_info->device_type == kHWDeviceTypeMTL) {
#ifdef BMF_LITE_ENABLE_METALBUFFER

        std::shared_ptr<MtlDeviceContext> mtl_context =
            std::make_shared<MtlDeviceContext>();
        int res = mtl_context->create_context();
        if (res < 0) {
            return res;
        }
        context = mtl_context;
        return BMF_LITE_StsOk;
#endif
    }
    if (create_info->device_type == kHWDeviceTypeNone) {
#ifdef BMF_LITE_ENABLE_CPUMEMORYBUFFER
        return BMF_LITE_StsOk;
#endif
    }
    return BMF_LITE_StsBadArg;
}

int HWDeviceContextManager::setHwDeviceContext(
    HardwareDeviceSetInfo *set_info,
    std::shared_ptr<HWDeviceContext> &context) {
    if (set_info == NULL) {
        // BMF_LITE_ERR("create info is NULL");
        return BMF_LITE_StsBadArg;
    }
    if (set_info->device_type == kHWDeviceTypeEGLCtx) {
#ifdef BMF_LITE_ENABLE_OPENGLTEXTUREBUFFER
        EGLContextInfo *egl_info = (EGLContextInfo *)(set_info->context_info);

        std::shared_ptr<EglHWDeviceContext> egl_context =
            std::make_shared<EglHWDeviceContext>(
                egl_info->egl_display, egl_info->egl_context,
                egl_info->egl_read_surface, egl_info->egl_draw_surface,
                set_info->owned);
        egl_context->setFreeFunc(set_info->free_func, set_info->user_data);
        context = egl_context;
        return BMF_LITE_StsOk;
#endif
    }
    if (set_info->device_type == kHWDeviceTypeMTL) {
#ifdef BMF_LITE_ENABLE_METALBUFFER
        void *mtl_info = (void *)(set_info->context_info);
        std::shared_ptr<MtlDeviceContext> mtl_context =
            std::make_shared<MtlDeviceContext>(mtl_info);
        return BMF_LITE_StsOk;
#endif
    }
    if (set_info->device_type == kHWDeviceTypeNone) {
#ifdef BMF_LITE_ENABLE_CPUMEMORYBUFFER
        return BMF_LITE_StsOk;
#endif
    }
    return BMF_LITE_StsBadArg;
}

} // namespace bmf_lite