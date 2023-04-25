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
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <hmp/imgproc.h>
#include <hmp/imgproc/image_seq.h>
#include <py_type_cast.h>
#include <py_utils.h>

namespace py = pybind11;

using namespace hmp;

Tensor tensor_from_numpy(const py::array& arr);
py::array tensor_to_numpy(const Tensor& tensor);


void imageBind(py::module &m)
{
#define VALUE(C, N) value("k" #N, C::N)

    py::enum_<ColorPrimaries>(m, "ColorPrimaries")
        .VALUE(ColorPrimaries, CP_RESERVED0)
        .VALUE(ColorPrimaries, CP_BT709)
        .VALUE(ColorPrimaries, CP_UNSPECIFIED)
        .VALUE(ColorPrimaries, CP_RESERVED)
        .VALUE(ColorPrimaries, CP_BT470M)
        .VALUE(ColorPrimaries, CP_BT470BG)
        .VALUE(ColorPrimaries, CP_SMPTE170M)
        .VALUE(ColorPrimaries, CP_SMPTE240M)
        .VALUE(ColorPrimaries, CP_FILM)
        .VALUE(ColorPrimaries, CP_BT2020)
        .VALUE(ColorPrimaries, CP_SMPTE428)
        .VALUE(ColorPrimaries, CP_SMPTEST428_1)
        .VALUE(ColorPrimaries, CP_SMPTE431)
        .VALUE(ColorPrimaries, CP_SMPTE432)
        .VALUE(ColorPrimaries, CP_EBU3213)
        .VALUE(ColorPrimaries, CP_JEDEC_P22)
        .VALUE(ColorPrimaries, CP_NB)
        .export_values();

    py::enum_<ColorTransferCharacteristic>(m, "ColorTransferCharacteristic")
        .VALUE(ColorTransferCharacteristic, CTC_RESERVED0)
        .VALUE(ColorTransferCharacteristic, CTC_BT709)
        .VALUE(ColorTransferCharacteristic, CTC_UNSPECIFIED)
        .VALUE(ColorTransferCharacteristic, CTC_RESERVED)
        .VALUE(ColorTransferCharacteristic, CTC_GAMMA22)
        .VALUE(ColorTransferCharacteristic, CTC_GAMMA28)
        .VALUE(ColorTransferCharacteristic, CTC_SMPTE170M)
        .VALUE(ColorTransferCharacteristic, CTC_SMPTE240M)
        .VALUE(ColorTransferCharacteristic, CTC_LINEAR)
        .VALUE(ColorTransferCharacteristic, CTC_LOG)
        .VALUE(ColorTransferCharacteristic, CTC_LOG_SQRT)
        .VALUE(ColorTransferCharacteristic, CTC_IEC61966_2_4)
        .VALUE(ColorTransferCharacteristic, CTC_BT1361_ECG)
        .VALUE(ColorTransferCharacteristic, CTC_IEC61966_2_1)
        .VALUE(ColorTransferCharacteristic, CTC_BT2020_10)
        .VALUE(ColorTransferCharacteristic, CTC_BT2020_12)
        .VALUE(ColorTransferCharacteristic, CTC_SMPTE2084)
        .VALUE(ColorTransferCharacteristic, CTC_SMPTEST2084)
        .VALUE(ColorTransferCharacteristic, CTC_SMPTE428)
        .VALUE(ColorTransferCharacteristic, CTC_SMPTEST428_1)
        .VALUE(ColorTransferCharacteristic, CTC_ARIB_STD_B67)
        .VALUE(ColorTransferCharacteristic, CTC_NB)
        .export_values();

    py::enum_<ColorSpace>(m, "ColorSpace")
        .VALUE(ColorSpace, CS_RGB)
        .VALUE(ColorSpace, CS_BT709)
        .VALUE(ColorSpace, CS_UNSPECIFIED)
        .VALUE(ColorSpace, CS_RESERVED)
        .VALUE(ColorSpace, CS_FCC)
        .VALUE(ColorSpace, CS_BT470BG)
        .VALUE(ColorSpace, CS_SMPTE170M)
        .VALUE(ColorSpace, CS_SMPTE240M)
        .VALUE(ColorSpace, CS_YCGCO)
        .VALUE(ColorSpace, CS_YCOCG)
        .VALUE(ColorSpace, CS_BT2020_NCL)
        .VALUE(ColorSpace, CS_BT2020_CL)
        .VALUE(ColorSpace, CS_SMPTE2085)
        .VALUE(ColorSpace, CS_CHROMA_DERIVED_NCL)
        .VALUE(ColorSpace, CS_CHROMA_DERIVED_CL)
        .VALUE(ColorSpace, CS_ICTCP)
        .VALUE(ColorSpace, CS_NB)
        .export_values();

    py::enum_<ColorRange>(m, "ColorRange")
        .VALUE(ColorRange, CR_UNSPECIFIED)
        .VALUE(ColorRange, CR_MPEG)
        .VALUE(ColorRange, CR_JPEG)
        .VALUE(ColorRange, CR_NB)
        .export_values();

    py::enum_<PixelFormat>(m, "PixelFormat")
#define DEF_VALUE(N) .VALUE(PixelFormat, N)
        HMP_FORALL_PIXEL_FORMATS(DEF_VALUE)
#undef DEF_VALUE
        .export_values();


    py::enum_<ChannelFormat>(m, "ChannelFormat")
        .value("kNCHW", kNCHW)
        .value("kNHWC", kNHWC)
        .export_values();

    py::enum_<ImageFilterMode>(m, "ImageFilterMode")
        .value("kNearest", ImageFilterMode::Nearest)
        .value("kBilinear", ImageFilterMode::Bilinear)
        .value("kBicubic", ImageFilterMode::Bicubic)
        .export_values();

    py::enum_<ImageRotationMode>(m, "ImageRotationMode")
        .value("kRotate0", ImageRotationMode::Rotate0)
        .value("kRotate90", ImageRotationMode::Rotate90)
        .value("kRotate180", ImageRotationMode::Rotate180)
        .value("kRotate270", ImageRotationMode::Rotate270)
        .export_values();

    py::enum_<ImageAxis>(m, "ImageAxis")
        .value("kHorizontal", ImageAxis::Horizontal)
        .value("kVertical", ImageAxis::Vertical)
        .value("kHorizontalAndVertical", ImageAxis::HorizontalAndVertical)
        .export_values();

    py::class_<ColorModel>(m, "PixelColorModel")
        .def(py::init<ColorSpace, ColorRange, ColorPrimaries, ColorTransferCharacteristic>())
        .def(py::init<ColorSpace, ColorRange>(),
            py::arg("cs"), py::arg("cr")=CR_UNSPECIFIED)
        .def(py::init<ColorPrimaries, ColorTransferCharacteristic>(),
            py::arg("cp"), py::arg("ctc")=CTC_UNSPECIFIED)
        .def_property_readonly("space", &ColorModel::space)
        .def_property_readonly("range", &ColorModel::range)
        .def_property_readonly("primaries", &ColorModel::primaries)
        .def_property_readonly("transfer_characteristic", &ColorModel::transfer_characteristic)
    ;

    py::class_<PixelInfo>(m, "PixelInfo")
        .def(py::init<PixelFormat, ColorModel>(),
            py::arg("format"), py::arg("color_model")=ColorModel())
        .def(py::init<PixelFormat, ColorSpace, ColorRange>(),
            py::arg("format"), py::arg("cs"), py::arg("cr")=CR_UNSPECIFIED)
        .def(py::init<PixelFormat, ColorPrimaries, ColorTransferCharacteristic>(),
            py::arg("format"), py::arg("cp"), py::arg("ctc")=CTC_UNSPECIFIED)
        .def_property_readonly("format", &PixelInfo::format)
        .def_property_readonly("color_model", &PixelInfo::color_model)
        .def_property_readonly("space", &PixelInfo::space)
        .def_property_readonly("range", &PixelInfo::range)
        .def_property_readonly("primaries", &PixelInfo::primaries)
        .def_property_readonly("transfer_characteristic", &PixelInfo::transfer_characteristic)
        .def("infer_space", &PixelInfo::infer_space)
    ;

    py::class_<PixelFormatDesc>(m, "PixelFormatDesc")
        .def(py::init<PixelFormat>())
        .def("defined", &PixelFormatDesc::defined)
        .def("channels", &PixelFormatDesc::channels, py::arg("plane")=0)
        .def("infer_width", &PixelFormatDesc::infer_width,
            py::arg("width"), py::arg("plane")=0)
        .def("infer_height", &PixelFormatDesc::infer_height,
            py::arg("height"), py::arg("plane")=0)
        .def("infer_nitems", (int(PixelFormatDesc::*)(int, int) const)&PixelFormatDesc::infer_nitems,
            py::arg("width"), py::arg("height"))
        .def("infer_nitems", (int(PixelFormatDesc::*)(int, int, int) const)&PixelFormatDesc::infer_nitems,
            py::arg("width"), py::arg("height"), py::arg("plane"))
        .def_property_readonly("nplanes", &PixelFormatDesc::nplanes)
        .def_property_readonly("dtype", &PixelFormatDesc::dtype)
        .def_property_readonly("format", &PixelFormatDesc::format)
    ;


    py::class_<Frame>(m, "Frame")
        .def(py::init<TensorList, const PixelInfo&>(),
             py::arg("planes"), py::arg("pix_info"))
        .def(py::init<TensorList, int, int, const PixelInfo&>(),
             py::arg("planes"), py::arg("width"), py::arg("height"), py::arg("pix_info"))
        .def(py::init<Tensor,  const PixelInfo&>(),
             py::arg("plane"), py::arg("pix_info"))
        .def(py::init([](int width, int height, const PixelInfo& pix_info, const py::object &obj){
            return Frame(width, height, pix_info, parse_device(obj));
        }), py::arg("width"), py::arg("height"), py::arg("pix_info"), py::arg("device")=kCPU)
        .def("__repr__", (std::string(*)(const Frame&))&stringfy)
        .def("format", &Frame::format)
        .def("pix_info", &Frame::pix_info)
        .def("pix_desc", &Frame::pix_desc)
        .def("width", &Frame::width)
        .def("height", &Frame::height)
        .def("dtype", &Frame::dtype)
        .def("device", &Frame::device)
        .def("plane", (Tensor&(Frame::*)(int64_t))&Frame::plane)
        .def("nplanes", &Frame::nplanes)
        .def("data", (TensorList&(Frame::*)())&Frame::data)
        .def("to", (Frame(Frame::*)(const Device&, bool) const)&Frame::to, py::arg("device"), py::arg("non_blocking")=false)
        .def("to", (Frame(Frame::*)(DeviceType, bool) const)&Frame::to, py::arg("device"), py::arg("non_blocking")=false)
        .def("to", [](const Frame &self, const std::string &device, bool non_blocking){
            return self.to(device);
        }, py::arg("device"), py::arg("non_blocking")=false)
        .def("copy_", &Frame::copy_)
        .def("clone", &Frame::clone)
        .def("crop", &Frame::crop, py::arg("left"), py::arg("top"), py::arg("width"), py::arg("height"))
        .def("reformat", &Frame::reformat, py::arg("pix_info"))
        .def("numpy", [](const Frame &frame){
            py::list arr_list;
            for(auto &tensor : frame.data()){
              arr_list.append(tensor_to_numpy(tensor));
            }
            return arr_list;
        })
    ;

    py::class_<FrameSeq>(m, "FrameSeq")
        .def(py::init<TensorList, const PixelInfo&>())
        .def("__repr__", (std::string(*)(const FrameSeq&))&stringfy)
        .def("format", &FrameSeq::format)
        .def("pix_info", &FrameSeq::pix_info)
        .def("pix_desc", &FrameSeq::pix_desc)
        .def("batch", &FrameSeq::batch)
        .def("width", &FrameSeq::width)
        .def("height", &FrameSeq::height)
        .def("plane", &FrameSeq::plane)
        .def("nplanes", &FrameSeq::nplanes)
        .def("data", &FrameSeq::data)
        .def("dtype", &FrameSeq::dtype)
        .def("device", &FrameSeq::device)
        .def("to", (FrameSeq(FrameSeq::*)(const Device&, bool) const)&FrameSeq::to, py::arg("device"), py::arg("non_blocking")=false)
        .def("to", (FrameSeq(FrameSeq::*)(DeviceType, bool) const)&FrameSeq::to, py::arg("device"), py::arg("non_blocking")=false)
        .def("to", [](const FrameSeq &self, const std::string &device, bool non_blocking){
            return self.to(device);
        }, py::arg("device"), py::arg("non_blocking")=false)
        .def("__getitem__", [](FrameSeq &self, int64_t index){ return self[index]; })
        .def("slice", &FrameSeq::slice, py::arg("start"), py::arg("end")=py::none())
        .def("crop", &FrameSeq::crop, py::arg("left"), py::arg("top"), py::arg("width"), py::arg("height"))
        .def("reformat", &FrameSeq::reformat, py::arg("pix_info"))
        .def("resize", &FrameSeq::resize, 
            py::arg("width"), py::arg("height"), py::arg("mode")=ImageFilterMode::Bilinear)
        .def("rotate", &FrameSeq::rotate, py::arg("mode"))
        .def("mirror", &FrameSeq::mirror, py::arg("axis")=ImageAxis::Horizontal)
    ;


    //
    m.def("concat", (FrameSeq(*)(const std::vector<Frame>&))&concat);
    m.def("concat", (FrameSeq(*)(const std::vector<FrameSeq>&))&concat);

    //
    auto img = m.def_submodule("img");
    img.def("yuv_to_rgb", 
        (Tensor&(*)(Tensor&, const TensorList&, const PixelInfo&, ChannelFormat))&img::yuv_to_rgb,
        py::arg("dst"), py::arg("src"), py::arg("pix_info"), py::arg("cformat")=kNCHW);
    img.def("yuv_to_rgb", 
        (Tensor(*)(const TensorList&, const PixelInfo&, ChannelFormat))&img::yuv_to_rgb,
        py::arg("src"), py::arg("pix_info"), py::arg("cformat")=kNCHW);

    img.def("rgb_to_yuv",
        (TensorList&(*)(TensorList&, const Tensor&, const PixelInfo&, ChannelFormat))&img::rgb_to_yuv,
        py::arg("dst"), py::arg("src"), py::arg("pix_info"), py::arg("cformat")=kNCHW);
    img.def("rgb_to_yuv",
        (TensorList(*)(const Tensor&, const PixelInfo&, ChannelFormat))&img::rgb_to_yuv,
        py::arg("src"), py::arg("pix_info"), py::arg("cformat")=kNCHW);

    img.def("yuv_resize", &img::yuv_resize);
    img.def("yuv_rotate", &img::yuv_rotate);
    img.def("yuv_mirror", &img::yuv_mirror);

    img.def("resize", (Tensor&(*)(Tensor&, const Tensor&, ImageFilterMode, ChannelFormat))&img::resize,
        py::arg("dst"), py::arg("src"), 
        py::arg("mode") = ImageFilterMode::Bilinear,
        py::arg("format") = kNCHW
    );
    img.def("resize", (Tensor(*)(const Tensor&, int, int, ImageFilterMode, ChannelFormat))&img::resize,
        py::arg("src"), py::arg("width"), py::arg("height"),
        py::arg("mode") = ImageFilterMode::Bilinear,
        py::arg("format") = kNCHW
    );

    img.def("rotate", (Tensor&(*)(Tensor&, const Tensor&, ImageRotationMode, ChannelFormat))&img::rotate,
        py::arg("dst"), py::arg("src"),
        py::arg("mode")=ImageRotationMode::Rotate90,
        py::arg("format")=kNCHW
    );
    img.def("rotate", (Tensor(*)(const Tensor&, ImageRotationMode, ChannelFormat))&img::rotate,
        py::arg("src"),
        py::arg("mode")=ImageRotationMode::Rotate90,
        py::arg("format")=kNCHW
    );

    img.def("mirror", (Tensor&(*)(Tensor&, const Tensor&, ImageAxis, ChannelFormat))&img::mirror,
        py::arg("dst"), py::arg("src"),
        py::arg("axis")=ImageAxis::Vertical, py::arg("format")=kNCHW
    );
    img.def("mirror", (Tensor(*)(const Tensor&, ImageAxis, ChannelFormat))&img::mirror,
        py::arg("src"),
        py::arg("axis")=ImageAxis::Vertical, py::arg("format")=kNCHW
    );

    img.def("normalize", (Tensor&(*)(Tensor&, const Tensor&, const Tensor&, const Tensor&, ChannelFormat))&img::normalize,
        py::arg("dst"), py::arg("src"),
        py::arg("mean"), py::arg("std"), py::arg("format")=kNCHW
    );
    img.def("normalize", (Tensor(*)(const Tensor&, const Tensor&, const Tensor&, ChannelFormat))&img::normalize,
        py::arg("src"),
        py::arg("mean"), py::arg("std"), py::arg("format")=kNCHW
    );

    img.def("erode", 
        (Tensor&(*)(Tensor&, const Tensor&, const optional<Tensor>&, ChannelFormat))&img::erode, 
        py::arg("dst"), py::arg("src"), py::arg("kernel")=py::none(), 
        py::arg("format")=kNCHW);
    img.def("erode", (Tensor(*)(const Tensor&, const optional<Tensor>&, ChannelFormat))&img::erode, 
        py::arg("src"), py::arg("kernel")=py::none(), 
        py::arg("format")=kNCHW);
    
    img.def("dilate", 
        (Tensor&(*)(Tensor&, const Tensor&, const optional<Tensor>&, ChannelFormat))&img::dilate, 
        py::arg("dst"), py::arg("src"), py::arg("kernel")=py::none(), 
        py::arg("format")=kNCHW);
    img.def("dilate",
        (Tensor(*)(const Tensor&, const optional<Tensor>&, ChannelFormat))&img::dilate, 
        py::arg("src"), py::arg("kernel")=py::none(), 
        py::arg("format")=kNCHW);

    img.def("sobel", 
        (Tensor&(*)(Tensor&, const Tensor&, int64_t, int64_t, int64_t, const Scalar&, const Scalar&, ChannelFormat))&img::sobel,
        py::arg("dst"), py::arg("src"), py::arg("dx"), py::arg("dy"),
        py::arg("ksize")=3, py::arg("scale")=1, py::arg("delta")=0,
        py::arg("format")=kNCHW);
    img.def("sobel", 
        (Tensor(*)(const Tensor&, int64_t, int64_t, int64_t, const Scalar&, const Scalar&, ChannelFormat))&img::sobel,
        py::arg("src"), py::arg("dx"), py::arg("dy"),
        py::arg("ksize")=3, py::arg("scale")=1, py::arg("delta")=0,
        py::arg("format")=kNCHW);

    img.def("canny", 
        (Tensor&(*)(Tensor&, const Tensor&, const Scalar&, const Scalar&, int64_t, bool, ChannelFormat))&img::canny,
        py::arg("dst"), py::arg("src"), py::arg("low_thresh"), py::arg("high_thresh"),
        py::arg("aperture")=3, py::arg("l2_gradient")=false,
        py::arg("format")=kNCHW);
    img.def("canny", 
        (Tensor(*)(const Tensor&, const Scalar&, const Scalar&, int64_t, bool, ChannelFormat))&img::canny,
        py::arg("src"), py::arg("low_thresh"), py::arg("high_thresh"),
        py::arg("aperture")=3, py::arg("l2_gradient")=false,
        py::arg("format")=kNCHW);

    img.def("filter2d", 
        (Tensor&(*)(Tensor&, const Tensor&, const Tensor&, const Scalar&, ChannelFormat))&img::filter2d,
        py::arg("dst"), py::arg("src"), py::arg("kernel"),
        py::arg("delta")=0, py::arg("format")=kNCHW);
    img.def("filter2d", 
        (Tensor(*)(const Tensor&, const Tensor&, const Scalar&, ChannelFormat))&img::filter2d,
        py::arg("src"), py::arg("kernel"),
        py::arg("delta")=0, py::arg("format")=kNCHW);

    img.def("warp_perspective", 
        (Tensor&(*)(Tensor&, const Tensor&, const Tensor&, ImageFilterMode, ChannelFormat))&img::warp_perspective,
        py::arg("dst"), py::arg("src"), py::arg("M"),
        py::arg("mode")=kBicubic, py::arg("format")=kNCHW);
    img.def("warp_perspective", 
        (Tensor(*)(const Tensor&, int64_t, int64_t, const Tensor&, ImageFilterMode, ChannelFormat))&img::warp_perspective,
        py::arg("src"), py::arg("width"), py::arg("height"), py::arg("M"),
        py::arg("mode")=kBicubic, py::arg("format")=kNCHW);

    img.def("bilateral_filter",
        (Tensor&(*)(Tensor&, const Tensor&, int, const Scalar&, const Scalar&, ChannelFormat))&img::bilateral_filter,
        py::arg("dst"), py::arg("src"), py::arg("d"), 
        py::arg("sigma_color"), py::arg("sigma_space"), py::arg("format")=kNCHW);
    img.def("bilateral_filter",
        (Tensor(*)(const Tensor&, int, const Scalar&, const Scalar&, ChannelFormat))&img::bilateral_filter,
        py::arg("src"), py::arg("d"), 
        py::arg("sigma_color"), py::arg("sigma_space"), py::arg("format")=kNCHW);

    img.def("gaussian_blur",
        (Tensor&(*)(Tensor&, const Tensor&, int, int, const Scalar&, const Scalar&, ChannelFormat))&img::gaussian_blur,
        py::arg("dst"), py::arg("src"), py::arg("kx"), py::arg("ky"), 
        py::arg("sigma_x"), py::arg("sigma_y")=0, py::arg("format")=kNCHW);
    img.def("gaussian_blur",
        (Tensor(*)(const Tensor&, int, int, const Scalar&, const Scalar&, ChannelFormat))&img::gaussian_blur,
        py::arg("src"), py::arg("kx"), py::arg("ky"), 
        py::arg("sigma_x"), py::arg("sigma_y")=0, py::arg("format")=kNCHW);

    img.def("overlay", 
        (Tensor&(*)(Tensor&, const Tensor&, const Tensor&, const Tensor&))&img::overlay,
        py::arg("dst"), py::arg("src0"), py::arg("src1"), py::arg("alpha"));
    img.def("overlay", 
        (Tensor(*)(const Tensor&, const Tensor&, const Tensor&))&img::overlay,
        py::arg("src0"), py::arg("src1"), py::arg("alpha"));

    img.def("transfer", 
        (Tensor(*)(const Tensor&, const ChannelFormat&, const ChannelFormat&))&img::transfer,
        py::arg("src"), py::arg("src_format"), py::arg("dst_format"));

}

