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

#include <hmp/torch/torch.h>
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/convert_backend.h>

namespace bmf_sdk {

template <> struct OpaqueDataInfo<at::Tensor> {
    const static int key = OpaqueDataKey::kATTensor;
    static OpaqueData construct(const at::Tensor *tensor) {
        return std::make_shared<at::Tensor>(*tensor);
    }
};

class TorchConvertor : public Convertor {
  public:
    TorchConvertor() {}
    int media_cvt(VideoFrame &src, const MediaDesc &dp) override {
        try {
            if (!src.frame().pix_info().is_rgbx()) {
                BMFLOG(BMF_ERROR) << "tensor only support rgbx frame";
                return -1;
            }
            auto tensor = src.frame().data()[0];
            at::Tensor at = hmp::torch::tensor(tensor);
            at::Tensor *pat = new at::Tensor(at);
            src.private_attach<at::Tensor>(pat);

        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "convert to at::tensor err: " << e.what();
            return -1;
        }
        return 0;
    }

    int media_cvt_to_videoframe(VideoFrame &src, const MediaDesc &dp) override {
        try {
            if (!dp.pixel_format.has_value()) {
                BMFLOG(BMF_ERROR) << "VideoFrame format represented by the "
                                     "at::Tensor must be specified.";
                return -1;
            }

            const at::Tensor *pat = src.private_get<at::Tensor>();
            if (!pat) {
                BMFLOG(BMF_ERROR) << "private data is null, please use "
                                     "private_attach before call this api";
                return -1;
            }
            Tensor tensor = hmp::torch::from_tensor(*pat);
            VideoFrame vf(Frame(tensor, hmp::PixelInfo(dp.pixel_format())));
            vf.private_attach<at::Tensor>(pat);
            src = vf;

        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "convert to at::tensor err: " << e.what();
            return -1;
        }
        return 0;
    }
};

static Convertor *torch_convert = new TorchConvertor();
BMF_REGISTER_CONVERTOR(MediaType::kATTensor, torch_convert);
} // namespace bmf_sdk
