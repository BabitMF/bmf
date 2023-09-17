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
#ifndef BMF_MAXINE_MODULE_H
#define BMF_MAXINE_MODULE_H

#include <bmf/sdk/bmf.h> 
#include <stdexcept>
#include <cuda_runtime.h>
#include "opencv2/opencv.hpp"
#include "nvVideoEffects.h"
#include "nvCVImage.h"

USE_BMF_SDK_NS

// Set an OpenCV Mat image from parameters
inline void CVImageSet(cv::Mat *cvIm, int width, int height, int numComps, int compType, int compBytes, void* pixels, size_t rowBytes) {
  size_t pixBytes   = numComps * compBytes;
  size_t widthBytes = width    * pixBytes;
  cvIm->flags = cv::Mat::MAGIC_VAL + (CV_MAKETYPE(compType, numComps) & cv::Mat::TYPE_MASK);
  if (rowBytes == widthBytes)
    cvIm->flags |= cv::Mat::CONTINUOUS_FLAG;
  cvIm->step.p    = cvIm->step.buf;
  cvIm->step[0]   = rowBytes;
  cvIm->step[1]   = pixBytes;
  cvIm->dims      = 2;
  cvIm->size      = cv::MatSize(&cvIm->rows);
  cvIm->rows      = height;
  cvIm->cols      = width;
  cvIm->data      = (uchar*)pixels;
  cvIm->datastart = (uchar*)pixels;
  cvIm->datalimit = cvIm->datastart + rowBytes * height;
  cvIm->dataend   = cvIm->datalimit - rowBytes + widthBytes;
  cvIm->allocator = 0;
  cvIm->u         = 0;
}

// Wrap an NvCVImage in a cv::Mat
inline void CVWrapperForNvCVImage(const NvCVImage *nvcvIm, cv::Mat *cvIm) {
  static const char cvType[] = { 7, 0, 2, 3, 7, 7, 4, 5, 7, 7, 6 };
  CVImageSet(cvIm, nvcvIm->width, nvcvIm->height, nvcvIm->numComponents, cvType[(int)nvcvIm->componentType], nvcvIm->componentBytes, nvcvIm->pixels, nvcvIm->pitch);
}

// Wrap a cv::Mat in an NvCVImage.
inline void NVWrapperForCVMat(const cv::Mat *cvIm, NvCVImage *nvcvIm) {
  static const NvCVImage_PixelFormat nvFormat[] = { NVCV_FORMAT_UNKNOWN, NVCV_Y, NVCV_YA, NVCV_BGR, NVCV_BGRA };
  static const NvCVImage_ComponentType nvType[] = { NVCV_U8, NVCV_TYPE_UNKNOWN, NVCV_U16, NVCV_S16, NVCV_S32, NVCV_F32,
                                                    NVCV_F64, NVCV_TYPE_UNKNOWN };
  nvcvIm->pixels         = cvIm->data;
  nvcvIm->width          = cvIm->cols;
  nvcvIm->height         = cvIm->rows;
  nvcvIm->pitch          = (int)cvIm->step[0];
  nvcvIm->pixelFormat    = nvFormat[cvIm->channels() <= 4 ? cvIm->channels() : 0];
  nvcvIm->componentType  = nvType[cvIm->depth() & 7];
  nvcvIm->bufferBytes    = 0;
  nvcvIm->deletePtr      = nullptr;
  nvcvIm->deleteProc     = nullptr;
  nvcvIm->pixelBytes     = (unsigned char)cvIm->step[1];
  nvcvIm->componentBytes = (unsigned char)cvIm->elemSize1();
  nvcvIm->numComponents  = (unsigned char)cvIm->channels();
  nvcvIm->planar         = NVCV_CHUNKY;
  nvcvIm->gpuMem         = NVCV_CPU;
  nvcvIm->reserved[0]    = 0;
  nvcvIm->reserved[1]    = 0;
}

inline __host__ static void checkMemUsed()
{
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    float used = total - avail;
    printf("global memory used: %f MB\n", float(used / 1024 / 1024));
}

struct Effect {
    std::string name_;
    NvVFX_Handle handle_;
    bool input_in_rgb_range_;
    std::vector<NvCVImage> gpu_buf_src_;
    std::vector<NvCVImage> gpu_buf_dst_;
};

class MaxineModule : public Module {
  public:
    MaxineModule(int node_id, JsonParam option);

    ~MaxineModule();

    virtual int process(Task &task);
  
  private:
    enum CompMode { compNone, compGreen, compWhite, compBG };

    int initialize();

    NvCV_Status check_scale_isotropy(const NvCVImage *src, const NvCVImage *dst);

    NvCV_Status vfx_setup(VideoFrame& vframe); 

    int set_bg_color(CompMode mode);

    NvCV_Status composition(NvCVImage &src, NvCVImage &dst, CompMode mode);

    std::vector<Effect> vfx_effects_;

	JsonParam option_;

    NvCVImage vfx_src_;

    NvCVImage vfx_dst_;

    NvCVImage vfx_tmp_;

    VideoFrame vframe_rgb24_out_;

    int num_effects_;

    std::string model_dir_;

    // Frame width
    int width_;

    // Frame height
    int height_;

	// for SuperRes and Upscale
    int resolution_;

    // for GreenScreen
    int max_num_streams_;

    // for GreenScreen
    CompMode comp_mode_;

    // for composition in GreenScreen
    unsigned char* bg_color_;

    // for bg file in GreenScreen
    cv::Mat bg_img_;
    cv::Mat resized_cropped_bg_img_;
    NvCVImage vfx_bg_;

    // for GreenScreen
    std::vector<NvVFX_StateObjectHandle> state_array_;

    // for GreenScreen
    NvVFX_StateObjectHandle* batch_of_states_;
    
    bool has_allocated_;

    cudaStream_t stream_;
};

#endif