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

#include "algorithm/bmf_algorithm.h"
#include "algorithm/bmf_video_frame.h"
#include "bmf_algorithm_impl.h"
#include "common/bmf_common.h"
#include "common/bmf_param.h"
#include <memory>
#include <vector>

namespace bmf_lite {

IAlgorithmInterface::~IAlgorithmInterface() {}

int IAlgorithmInterface::setParam(Param param) { return -1; }

int IAlgorithmInterface::processVideoFrame(VideoFrame videoframe, Param param) {
    return -1;
}

int IAlgorithmInterface::getVideoFrameOutput(VideoFrame &frame, Param &param) {
    return -1;
}

int IAlgorithmInterface::processMultiVideoFrame(
    std::vector<VideoFrame> videoframes, Param param) {
    return -1;
}

int IAlgorithmInterface::getMultiVideoFrameOutput(
    std::vector<VideoFrame> &videoframes, Param &param) {
    return -1;
}

int IAlgorithmInterface::getProcessProperty(Param &param) { return -1; }

int IAlgorithmInterface::setInputProperty(Param attr) { return -1; }

int IAlgorithmInterface::getOutputProperty(Param &attr) { return -1; }

IAlgorithm::IAlgorithm() { impl_ = new AlgorithmImpl(); }

IAlgorithm::~IAlgorithm() {
    if (impl_ != NULL) {
        delete impl_;
    }
}

int IAlgorithm::setParam(Param param) { return impl_->setParam(param); }

int IAlgorithm::processVideoFrame(VideoFrame videoframe, Param param) {
    return impl_->processVideoFrame(videoframe, param);
}

int IAlgorithm::getVideoFrameOutput(VideoFrame &videoframe, Param &param) {
    return impl_->getVideoFrameOutput(videoframe, param);
}

int IAlgorithm::processMultiVideoFrame(std::vector<VideoFrame> videoframes,
                                       Param param) {
    return impl_->processMultiVideoFrame(videoframes, param);
}

int IAlgorithm::getMultiVideoFrameOutput(std::vector<VideoFrame> &videoframes,
                                         Param &param) {
    return impl_->getMultiVideoFrameOutput(videoframes, param);
}

int IAlgorithm::getProcessProperty(Param &param) {
    return impl_->getProcessProperty(param);
}

int IAlgorithm::setInputProperty(Param param) {
    return impl_->setInputProperty(param);
}

int IAlgorithm::getOutputProperty(Param &param) {
    return impl_->getOutputProperty(param);
}

IAlgorithm *AlgorithmFactory::createAlgorithmInterface() {
    return new IAlgorithm();
}

void AlgorithmFactory::releaseAlgorithmInterface(IAlgorithm *instance) {
    delete instance;
}

} // namespace bmf_lite
