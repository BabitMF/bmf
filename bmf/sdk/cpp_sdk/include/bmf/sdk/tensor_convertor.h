#pragma once

#include <hmp/tensor.h>
#include <bmf/sdk/sdk_interface.h>
#include <bmf/sdk/log.h>
#include <bmf/sdk/convert_backend.h>

namespace bmf_sdk {

template <> struct OpaqueDataInfo<hmp::Tensor> {
    const static int key = OpaqueDataKey::kTensor;
    static OpaqueData construct(const hmp::Tensor *tensor) {
        return std::make_shared<Tensor>(*tensor);
    }
};

class TensorConvertor : public Convertor {
  public:
    TensorConvertor() {}
    int media_cvt(VideoFrame &src, const MediaDesc &dp) override {
        try {
            if (!src.frame().pix_info().is_rgbx()) {
                BMFLOG(BMF_ERROR) << "tensor only support rgbx frame";
                return -1;
            }
            auto tensor = src.frame().data()[0];
            hmp::Tensor *pt = new hmp::Tensor(tensor);
            src.private_attach<hmp::Tensor>(pt);

        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "convert to hmp::tensor err: " << e.what();
            return -1;
        }
        return 0;
    }

    int media_cvt_to_videoframe(VideoFrame &src, const MediaDesc &dp) override {
        try {
            if (!dp.pixel_format.has_value()) {
                BMFLOG(BMF_ERROR) << "VideoFrame format represented by the "
                                     "Tensor must be specified.";
                return -1;
            }

            const hmp::Tensor *pt = src.private_get<hmp::Tensor>();
            if (!pt) {
                BMFLOG(BMF_ERROR) << "private data is null, please use "
                                     "private_attach before call this api";
                return -1;
            }
            Tensor tensor = *pt;
            VideoFrame vf(Frame(tensor, hmp::PixelInfo(dp.pixel_format())));
            vf.private_attach<hmp::Tensor>(pt);
            src = vf;

        } catch (std::exception &e) {
            BMFLOG(BMF_ERROR) << "convert to hmp::tensor err: " << e.what();
            return -1;
        }
        return 0;
    }
};

static Convertor *tensor_convert = new TensorConvertor();
BMF_REGISTER_CONVERTOR(MediaType::kTensor, tensor_convert);

} // namespace bmf_sdk
