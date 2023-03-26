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
#include <hmp_capi.h>
#include <hmp/format.h>

thread_local std::string s_hmp_last_error;

#define HMP_PROTECT(...) 	            \
	try{                                \
        __VA_ARGS__                              \
    } catch(const std::exception &e){   \
        s_hmp_last_error = e.what();    \
        HMP_WRN("Exception: {}", e.what()); \
    }


using namespace hmp;



const char *hmp_last_error()
{
    return s_hmp_last_error.c_str();
}

/////// hmp_Scalar ///////////
hmp_Scalar hmp_scalar_float(double v)
{
    return new Scalar(v);
}

hmp_Scalar hmp_scalar_int(int64_t v)
{
    return new Scalar(v);
}

hmp_Scalar hmp_scalar_bool(bool v)
{
    return new Scalar(v);
}

void hmp_scalar_free(hmp_Scalar scalar)
{
    if(scalar){
        delete scalar;
    }
}


////////// hmp_Device //////////
int hmp_device_count(int device_type)
{
    HMP_PROTECT(
        return device_count(DeviceType(device_type));
    )
    return 0;
}



////////// hmp_Stream /////////
hmp_Stream hmp_stream_create(int device_type, uint64_t flags)
{
    HMP_PROTECT(
        return new Stream(create_stream((DeviceType)device_type, flags));
    )
    return nullptr;
}

void hmp_stream_free(hmp_Stream stream)
{
    if(stream){
        delete stream;
    }
}

bool hmp_stream_query(hmp_Stream stream)
{
    HMP_PROTECT(
        return stream->query();
    )
    return false;
}

void hmp_stream_synchronize(hmp_Stream stream)
{
    HMP_PROTECT(
        stream->synchronize();
    )
}

uint64_t hmp_stream_handle(const hmp_Stream stream)
{
    return stream->handle();
}

int  hmp_stream_device_type(const hmp_Stream stream)
{
    return (int)stream->device().type();
}

int  hmp_stream_device_index(const hmp_Stream stream)
{
    return stream->device().index();
}


void hmp_stream_set_current(const hmp_Stream stream)
{
    return set_current_stream(*stream);
}

hmp_Stream hmp_stream_current(int device_type)
{
    HMP_PROTECT(
        return new Stream(current_stream((DeviceType)device_type).value());
    )
    return nullptr;
}


hmp_StreamGuard hmp_stream_guard_create(hmp_Stream stream)
{
    HMP_PROTECT(
        return new StreamGuard(*stream);
    )
    return nullptr;
}

void hmp_stream_guard_free(hmp_StreamGuard guard)
{
    if(guard){
        delete guard;
    }
}


/////// hmp_Tensor ///////////

hmp_Tensor hmp_tensor_empty(const int64_t *shape, int ndim, int type, const char *device, bool pinned_memory)
{
    SizeArray vshape(shape, shape+ndim);
    auto options = TensorOptions((ScalarType)type)
                    .device(Device(device))
                    .pinned_memory(pinned_memory);
    HMP_PROTECT(
        return new Tensor(empty(vshape, options));
    );

    return nullptr;
}

hmp_Tensor hmp_tensor_arange(int64_t start, int64_t end, int64_t step, int type, const char *device, bool pinned_memory)
{
    auto options = TensorOptions((ScalarType)type)
                    .device(Device(device))
                    .pinned_memory(pinned_memory);
    HMP_PROTECT(
        return new Tensor(arange(start, end, step, options));
    )
    return nullptr;
}

void hmp_tensor_free(hmp_Tensor tensor)
{
    if(tensor){
        delete tensor;
    }
}


thread_local std::string s_tensor_stringfy_str; 
const char* hmp_tensor_stringfy(hmp_Tensor tensor, int *size)
{
    HMP_PROTECT(
        s_tensor_stringfy_str = stringfy(*tensor);
        *size = s_tensor_stringfy_str.size();
        return s_tensor_stringfy_str.c_str();
    )

    return nullptr;
}

void hmp_tensor_fill(hmp_Tensor tensor, hmp_Scalar value)
{
    HMP_PROTECT(
        fill(*tensor, *value);
    )
}

bool hmp_tensor_defined(hmp_Tensor tensor)
{
    HMP_PROTECT(
        return tensor->defined();
    )

    return false;
}

int64_t hmp_tensor_dim(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return tensor->dim();
    )
    return -1;
}

int64_t hmp_tensor_size(const hmp_Tensor tensor, int64_t dim)
{
    HMP_PROTECT(
        return tensor->size(dim);
    )
    return -1;
}


int64_t hmp_tensor_itemsize(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return tensor->itemsize();
    )
    return -1;
}

int64_t hmp_tensor_stride(const hmp_Tensor tensor, int64_t dim)
{
    HMP_PROTECT(
        return tensor->stride(dim);
    )
    return -1;
}

int64_t hmp_tensor_nitems(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return tensor->nitems();
    )
    return -1;
}

int64_t hmp_tensor_nbytes(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return tensor->nbytes();
    )
    return -1;
}

int hmp_tensor_dtype(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return (int)tensor->dtype();
    )
    return -1;
}

void* hmp_tensor_data(hmp_Tensor tensor)
{
    HMP_PROTECT(
        return tensor->unsafe_data();
    )
    return nullptr;
}

bool hmp_tensor_is_contiguous(hmp_Tensor tensor)
{
    HMP_PROTECT(
        return tensor->is_contiguous();
    )
    return false;
}

int hmp_tensor_device_type(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return (int)tensor->device_type();
    )
    return -1;
}

int hmp_tensor_device_index(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return tensor->device_index();
    )
    return -1;
}


hmp_Tensor hmp_tensor_clone(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return new Tensor(tensor->clone());
    )
    return nullptr;
}

hmp_Tensor hmp_tensor_alias(const hmp_Tensor tensor)
{
    HMP_PROTECT(
        return new Tensor(tensor->alias());
    )
    return nullptr;
}

hmp_Tensor hmp_tensor_view(const hmp_Tensor tensor, const int64_t *shape, int ndim)
{
    HMP_PROTECT(
        SizeArray vshape{shape, shape+ndim};
        return new Tensor(tensor->view(vshape));
    )
    return nullptr;
}

hmp_Tensor hmp_tensor_reshape(const hmp_Tensor tensor, const int64_t *shape, int ndim)
{
    HMP_PROTECT(
        SizeArray vshape{shape, shape+ndim};
        return new Tensor(tensor->reshape(vshape));
    )
    return nullptr;
}


hmp_Tensor hmp_tensor_slice(const hmp_Tensor tensor, int64_t dim,
                            int64_t start, int64_t end, int64_t step)
{
    HMP_PROTECT(
        return new Tensor(tensor->slice(dim, start, end, step));
    )
    return nullptr;
}


hmp_Tensor hmp_tensor_select(const hmp_Tensor tensor, int64_t dim, int64_t index)
{
    HMP_PROTECT(
        return new Tensor(tensor->select(dim, index));
    )
    return nullptr;
}


hmp_Tensor hmp_tensor_permute(const hmp_Tensor tensor, const int64_t *dims, int ndim)
{
    HMP_PROTECT(
        SizeArray vdims{dims, dims+ndim};
        return new Tensor(tensor->permute(vdims));
    )
    return nullptr;
}


hmp_Tensor hmp_tensor_squeeze(const hmp_Tensor tensor, int64_t dim)
{
    HMP_PROTECT(
        return new Tensor(tensor->squeeze(dim));
    )
    return nullptr;
}


hmp_Tensor hmp_tensor_unsqueeze(const hmp_Tensor tensor, int64_t dim)
{
    HMP_PROTECT(
        return new Tensor(tensor->unsqueeze(dim));
    )
    return nullptr;
}


hmp_Tensor hmp_tensor_to_device(const hmp_Tensor data, const char *device, bool non_blocking)
{
    HMP_PROTECT(
        return new Tensor(data->to(Device(device), non_blocking));
    )
    return nullptr;
}

hmp_Tensor hmp_tensor_to_dtype(const hmp_Tensor data, int dtype)
{
    HMP_PROTECT(
        return new Tensor(data->to((ScalarType)dtype));
    )
    return nullptr;
}

void hmp_tensor_copy_from(hmp_Tensor data, const hmp_Tensor from)
{
    data->copy_(*from);
}

/////////////////// hmp_ColorModel ////////////
hmp_ColorModel hmp_color_model(int cs, int cr, int cp, int ctc)
{
    HMP_PROTECT(
        return new ColorModel((ColorSpace)cs, (ColorRange)cr,
                              (ColorPrimaries)cp, (ColorTransferCharacteristic)ctc);
    )
    return nullptr;
}

void hmp_color_model_free(hmp_ColorModel cm)
{
    if(cm){
        delete cm;
    }
}

int hmp_color_model_space(const hmp_ColorModel cm)
{
    return (int)cm->space();
}

int hmp_color_model_range(const hmp_ColorModel cm)
{
    return (int)cm->range();
}

int hmp_color_model_primaries(const hmp_ColorModel cm)
{
    return (int)cm->primaries();
}

int hmp_color_model_ctc(const hmp_ColorModel cm)
{
    return (int)cm->transfer_characteristic();
}


hmp_PixelInfo hmp_pixel_info(int format, const hmp_ColorModel cm)
{
    HMP_PROTECT(
        return new PixelInfo((PixelFormat)format, *cm);
    )
    return nullptr;
}

hmp_PixelInfo hmp_pixel_info_v1(int format, int cs, int cr)
{
    HMP_PROTECT(
        return new PixelInfo((PixelFormat)format, (ColorSpace)cs, (ColorRange)cr);
    )
    return nullptr;
}

hmp_PixelInfo hmp_pixel_info_v2(int format, int cp, int ctc)
{
    HMP_PROTECT(
        return new PixelInfo((PixelFormat)format,
                            (ColorPrimaries)cp, (ColorTransferCharacteristic)ctc);
    )
    return nullptr;
}

void hmp_pixel_info_free(hmp_PixelInfo pix_info)
{
    if(pix_info){
        delete pix_info;
    }
}

int hmp_pixel_info_format(const hmp_PixelInfo pix_info)
{
    return (int)pix_info->format();
}

int hmp_pixel_info_space(const hmp_PixelInfo pix_info)
{
    return (int)pix_info->space();
}

int hmp_pixel_info_range(const hmp_PixelInfo pix_info)
{
    return (int)pix_info->range();
}

int hmp_pixel_info_primaries(const hmp_PixelInfo pix_info)
{
    return (int)pix_info->primaries();
}

int hmp_pixel_info_ctc(const hmp_PixelInfo pix_info)
{
    return (int)pix_info->transfer_characteristic();
}

int hmp_pixel_info_infer_space(const hmp_PixelInfo pix_info)
{
    return (int)pix_info->infer_space();
}

const hmp_ColorModel hmp_pixel_info_color_model(const hmp_PixelInfo pix_info)
{
    return (const hmp_ColorModel)&pix_info->color_model();
}

bool hmp_pixel_info_is_rgbx(const hmp_PixelInfo pix_info)
{
    return pix_info->is_rgbx();
}

thread_local std::string s_pixel_info_stringfy_str;
const char *hmp_pixel_info_stringfy(const hmp_PixelInfo pix_info, int *size)
{
    s_pixel_info_stringfy_str = stringfy(*pix_info);
    *size = s_pixel_info_stringfy_str.size();
    return s_pixel_info_stringfy_str.c_str();
}


/////////////////// hmp_Frame /////////////////
hmp_Frame hmp_frame_make(int width, int height, const hmp_PixelInfo pix_info, const char *device)
{
    HMP_PROTECT(
        return new Frame(width, height, *pix_info, Device(device));
    )
    return nullptr;
}

hmp_Frame hmp_frame_from_data(hmp_Tensor *data, int size, const hmp_PixelInfo pix_info)
{
    TensorList vdata;
    for(int i = 0; i < size; ++i){
        vdata.push_back(*data[i]);
    }
    HMP_PROTECT(
        return new Frame(vdata, *pix_info);
    )
    return nullptr;
}

hmp_Frame hmp_frame_from_data_v1(hmp_Tensor *data, int size, int width, int height, const hmp_PixelInfo pix_info)
{
    TensorList vdata;
    for(int i = 0; i < size; ++i){
        vdata.push_back(*data[i]);
    }
    HMP_PROTECT(
        return new Frame(vdata, width, height, *pix_info);
    )
    return nullptr;
}

void hmp_frame_free(hmp_Frame frame)
{
    if(frame){
        delete frame;
    }
}

bool hmp_frame_defined(const hmp_Frame frame)
{
    return *frame;
}

const hmp_PixelInfo hmp_frame_pix_info(const hmp_Frame frame)
{
    return (const hmp_PixelInfo)&frame->pix_info();
}

int hmp_frame_format(const hmp_Frame frame)
{
    return (int)frame->format();
}

int hmp_frame_width(const hmp_Frame frame)
{
    return frame->width();
}

int hmp_frame_height(const hmp_Frame frame)
{
    return frame->height();
}

int hmp_frame_dtype(const hmp_Frame frame)
{
    return (int)frame->dtype();
}

int hmp_frame_device_type(const hmp_Frame frame)
{
    return (int)frame->device().type();
}

int hmp_frame_device_index(const hmp_Frame frame)
{
    return frame->device().index();
}

int64_t hmp_frame_nplanes(const hmp_Frame frame)
{
    return frame->nplanes();
}

const hmp_Tensor hmp_frame_plane(const hmp_Frame frame, int64_t p)
{
    return &frame->plane(p);
}

hmp_Frame hmp_frame_to_device(const hmp_Frame frame, const char *device, bool non_blocking)
{
    HMP_PROTECT(
        return new Frame(frame->to(Device(device), non_blocking));
    )
    return nullptr;
}

void hmp_frame_copy_from(hmp_Frame self, const hmp_Frame from)
{
    HMP_PROTECT(
        self->copy_(*from);
    )
}

hmp_Frame hmp_frame_clone(const hmp_Frame frame)
{
    HMP_PROTECT(
        return new Frame(frame->clone());
    )
    return nullptr;
}

hmp_Frame hmp_frame_crop(const hmp_Frame frame, int left, int top, int width, int height)
{
    HMP_PROTECT(
        return new Frame(frame->crop(left, top, width, height));
    )
    return nullptr;
}

hmp_Image hmp_frame_to_image(const hmp_Frame frame, int cformat)
{
    HMP_PROTECT(
        return new Image(frame->to_image((ChannelFormat)cformat));
    )
    return nullptr;
}

hmp_Frame hmp_frame_from_image(const hmp_Image image, const hmp_PixelInfo pix_info)
{
    HMP_PROTECT(
        return new Frame(Frame::from_image(*image, *pix_info));
    )
    return nullptr;
}

thread_local std::string s_frame_stringfy_str;
const char* hmp_frame_stringfy(const hmp_Frame frame, int *size)
{
    s_frame_stringfy_str = stringfy(*frame);
    *size = s_frame_stringfy_str.size();
    return s_frame_stringfy_str.c_str();
}


/////////////////// hmp_Image /////////////////
hmp_Image hmp_image_make(int width, int height, int channels, int cformat, 
				int type, const char *device, bool pinned_memory)
{
    auto options = TensorOptions((ScalarType)type)
                    .device(Device(device))
                    .pinned_memory(pinned_memory);
    HMP_PROTECT(
        return new Image(width, height, channels, (ChannelFormat)cformat, options);
    )
    return nullptr;
}

hmp_Image hmp_image_from_data(const hmp_Tensor data, int cformat)
{
    HMP_PROTECT(
        return new Image(*data, (ChannelFormat)cformat);
    )
    return nullptr;
}

hmp_Image hmp_image_from_data_v1(const hmp_Tensor data, int cformat, const hmp_ColorModel cm)
{
    HMP_PROTECT(
        return new Image(*data, (ChannelFormat)cformat, *cm);
    )
    return nullptr;
}

void hmp_image_free(hmp_Image image)
{
    if(image){
        delete image;
    }
}

bool hmp_image_defined(const hmp_Image image)
{
    return *image;
}

int hmp_image_format(const hmp_Image image)
{
    return (int)image->format(); 
}

void hmp_image_set_color_model(hmp_Image image, const hmp_ColorModel cm)
{
    image->set_color_model(*cm);
}

const hmp_ColorModel hmp_image_color_model(const hmp_Image image)
{
    return (const hmp_ColorModel)&image->color_model();
}

int hmp_image_wdim(const hmp_Image image)
{
    return image->wdim();
}

int hmp_image_hdim(const hmp_Image image)
{
    return image->hdim();
}

int hmp_image_cdim(const hmp_Image image)
{
    return image->cdim();
}

int hmp_image_width(const hmp_Image image)
{
    return image->width();
}

int hmp_image_height(const hmp_Image image)
{
    return image->height();
}

int hmp_image_nchannels(const hmp_Image image)
{
    return image->nchannels();
}

int hmp_image_dtype(const hmp_Image image)
{
    return (int)image->dtype();
}

int hmp_image_device_type(const hmp_Image image)
{
    return (int)image->device().type();
}

int hmp_image_device_index(const hmp_Image image)
{
    return image->device().index();
}

const hmp_Tensor hmp_image_data(const hmp_Image image)
{
    return (const hmp_Tensor)&image->data();
}

hmp_Image hmp_image_to_device(const hmp_Image image, const char *device, bool non_blocking)
{
    HMP_PROTECT(
        return new Image(image->to(Device(device), non_blocking));
    )
    return nullptr;
}

hmp_Image hmp_image_to_dtype(const hmp_Image image, int dtype)
{
    HMP_PROTECT(
        return new Image(image->to((ScalarType)dtype));
    )
    return nullptr;
}

void hmp_image_copy_from(hmp_Image image, const hmp_Image from)
{
    HMP_PROTECT(
        image->copy_(*from);
    )
}

hmp_Image hmp_image_clone(const hmp_Image image)
{
    HMP_PROTECT(
        return new Image(image->clone());
    )
    return nullptr;
}

hmp_Image hmp_image_crop(const hmp_Image image, int left, int top, int width, int height)
{
    HMP_PROTECT(
        return new Image(image->crop(left, top, width, height));
    )
    return nullptr;
}

hmp_Image hmp_image_select(const hmp_Image image, int channel)
{
    HMP_PROTECT(
        return new Image(image->select(channel));
    )
    return nullptr;
}

thread_local std::string s_image_stringfy_str;
const char* hmp_image_stringfy(const hmp_Image image, int *size)
{
    s_image_stringfy_str = stringfy(*image);
    *size = s_image_stringfy_str.size();
    return s_image_stringfy_str.c_str();
}