#include <queue>
#include <stdlib.h>
#include <string.h>

#include "egl_helper.h"
#include <dirent.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
int g_frame_index = 0;
#include "bmf_lite.h"

int saveVideoFrame(std::shared_ptr<egl::EGLHelper> egl_api,
                   std::string image_path, bmf_lite::VideoFrame &video_frame) {
    int width = video_frame.buffer()->width();
    int height = video_frame.buffer()->height();
    int output_texture = (long)(video_frame.buffer()->data());
    std::cout << "save video frame width:" << width << " height:" << height
              << " texture id:" << output_texture << std::endl;
    unsigned char *rgba_data = (unsigned char *)malloc(width * height * 4);
    egl_api->copy_from_texture(output_texture, rgba_data, width, height);
    stbi_write_png(image_path.c_str(), width, height, 4, rgba_data, 0);
    stbi_image_free(rgba_data);
    return 0;
}
int createVideoFrame(std::shared_ptr<egl::EGLHelper> egl_api,
                     std::string image_path,
                     bmf_lite::VideoFrame &video_frame) {
    int width, height, channel;
    unsigned char *image_data =
        stbi_load(image_path.c_str(), &width, &height, &channel, 0);
    unsigned char *image_data_rgba = nullptr;
    if (channel == 3) {
        image_data_rgba = (unsigned char *)malloc(width * height * 4);
        for (int i = 0; i < width * height; i++) {
            image_data_rgba[4 * i] = image_data[3 * i];
            image_data_rgba[4 * i + 1] = image_data[3 * i + 1];
            image_data_rgba[4 * i + 2] = image_data[3 * i + 2];
            image_data_rgba[4 * i + 3] = 255;
        }
    } else if (channel == 4) {
        image_data_rgba = image_data;
    } else {
        std::cerr << "createVideoFrame fail, input image channel is too small "
                  << std::endl;
        return -1;
    }
    int texture_id = egl_api->create_texture(width, height);
    egl_api->copy_to_texture(texture_id, image_data_rgba, width, height);
    int size = width * height * 4;
    bmf_lite::HardwareDeviceSetInfo set_info_new;
    bmf_lite::HWDeviceType device_type = bmf_lite::kHWDeviceTypeEGLCtx;
    set_info_new.device_type = device_type;
    bmf_lite::EGLContextInfo egl_context_info_create{
        egl_api->m_eglDisplay, egl_api->m_eglContext, egl_api->m_eglSurface,
        egl_api->m_eglSurface};
    set_info_new.context_info = &egl_context_info_create;
    set_info_new.owned = 0;
    std::shared_ptr<bmf_lite::HWDeviceContext> device_context_new;
    bmf_lite::HWDeviceContextManager::setHwDeviceContext(&set_info_new,
                                                         device_context_new);
    bmf_lite::HardwareDataInfo data_info = {
        bmf_lite::MemoryType::kOpenGLTexture2d, bmf_lite::GLES_TEXTURE_RGBA, 0};
    std::shared_ptr<bmf_lite::VideoBuffer> video_buffer;
    bmf_lite::VideoBufferManager::createTextureVideoBufferFromExistingData(
        (void *)texture_id, width, height, &data_info, device_context_new, NULL,
        video_buffer);
    video_frame = bmf_lite::VideoFrame(video_buffer);
    if (channel == 3 && image_data_rgba != nullptr) {
        free(image_data_rgba);
        image_data_rgba = nullptr;
    }

    return 0;
}

int createOesVideoFrame(std::shared_ptr<egl::EGLHelper> egl_api,
                        std::string image_path,
                        bmf_lite::VideoFrame &video_frame) {
    int width, height, channel;
    unsigned char *image_data =
        stbi_load(image_path.c_str(), &width, &height, &channel, 0);
    unsigned int hold_texture;
    EGLImage mEGLImage;
    int texture_id =
        egl_api->create_oes_texture(width, height, mEGLImage, hold_texture);
    egl_api->copy_to_texture(hold_texture, image_data, width, height);
    return 0;
}

int test_algorithm(int algorithmType = 1) {
    std::shared_ptr<egl::EGLHelper> egl_api_ =
        std::make_shared<egl::EGLHelper>(1280, 720);
    if (!egl_api_->initilized()) {
        throw std::runtime_error("EGL init failed\n");
    }
    std::cout << "createAlgorithmInterface" << std::endl;
    bmf_lite::IAlgorithm *algorithm =
        bmf_lite::AlgorithmFactory::createAlgorithmInterface();
    bmf_lite::VideoFrame video_frame;
    bmf_lite::Param init_param;
    bmf_lite::Param process_param;
    bmf_lite::Param output_param;
    bmf_lite::VideoFrame output_video_frame;
    std::string result_img_path = "denoise.jpg";
    std::string input_img_path = "test.jpg";
    init_param.setInt("change_mode", 6);
    if (algorithmType == 0) {
        init_param.setString("instance_id", "super_resolution");
        init_param.setInt("scale_mode", 0); // for super_resolution only
        init_param.setInt("sharp_levels", 0);
        process_param.setInt("sharp_level", 0);
        process_param.setInt("scale_mode", 0);
        init_param.setString("weight_path", "");
        result_img_path = "super_resolution.jpg";
    } else if (algorithmType == 3) {
        std::string positive_prompt_en = "cute,girl";
        std::string negative_prompt_en = "";
        init_param.setString("instance_id", "tex2pic");
        init_param.setString("qnn_htp_library_path", "libQnnHtp.so");
        init_param.setString("qnn_system_library_path", "libQnnSystem.so");
        process_param.setString("positive_prompt_en", positive_prompt_en);
        process_param.setString("negative_prompt_en",
                                negative_prompt_en); // for tex2pic only
        input_img_path = "test-canny.png";
    } else {
        init_param.setString("instance_id", "denoise"); // default denoise
    }
    init_param.setInt("algorithm_version", 0);
    init_param.setInt(
        "algorithm_type",
        algorithmType); // 0 super_resolution, 1 denoise, 3 tex2pic
    init_param.setInt("backend", 3);
    init_param.setInt("process_mode", 0);
    init_param.setInt("max_width", 1920);
    init_param.setInt("max_height", 1080);
    init_param.setString("license_module_name", "");
    init_param.setString("program_cache_dir", "");
    int init_result = algorithm->setParam(init_param);
    if (algorithmType == 3 && init_result != 0) {
        std::cerr << "this devices is not support tex2pic init_result "
                  << init_result << std::endl;
        return -1;
    }

    std::cout << "init result:" << init_result << std::endl;
    int ret = createVideoFrame(egl_api_, input_img_path, video_frame);
    if (ret != 0) {
        std::cerr << "the format of the input picture is incorrect"
                  << std::endl;
        return -1;
    }
    process_param.setInt("process_mode", 0);
    saveVideoFrame(egl_api_, "backup.jpg", video_frame);
    int process_result =
        algorithm->processVideoFrame(video_frame, process_param);
    std::cout << "processVideoFrame result = " << process_result << std::endl;
    algorithm->getVideoFrameOutput(output_video_frame, output_param);
    saveVideoFrame(egl_api_, result_img_path, output_video_frame);
    bmf_lite::AlgorithmFactory::releaseAlgorithmInterface(algorithm);
    return 0;
}

int main(int argc, char *argv[]) {
    std::cout << "test super_resolution start" << std::endl;
    test_algorithm(0);
    std::cout << "test super_resolution end" << std::endl;
    std::cout << "test denoise start" << std::endl;
    test_algorithm(1);
    std::cout << "test denoise end" << std::endl;
    // std::cout << "test tex2pic start" << std::endl;
    // test_algorithm(3);
    // If you need to run the vincennes chart sample through the cpp
    // test program, you need the 8gen3 chip. In addition, you need to
    // configure the runtime environment by referring to the way the app
    // runs, and change the library path in the code, code path:
    // bmf_lite/src/algorithm/contrib_modules/QnnControlNet/QnnControlNet_algorithm.cpp
    // std::cout << "test tex2pic end" << std::endl;
    return 0;
}