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

extern "C"{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#include <libavutil/opt.h>

}
#include <hmp/ffmpeg/ffmpeg.h>

namespace hmp{
namespace ffmpeg{

static std::string AVErr2Str(int rc)
{
    char av_error[AV_ERROR_MAX_STRING_SIZE] = { 0 };
    av_make_error_string(av_error, AV_ERROR_MAX_STRING_SIZE, rc);
    return std::string(av_error);
}




}} //