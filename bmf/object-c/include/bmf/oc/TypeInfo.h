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
#include <bmf/sdk/bmf_type_info.h>
#include <hmp/oc/CV.h>
#include <hmp/oc/Metal.h>

namespace bmf_sdk {
namespace metal {

using hmp::metal::Device;
using hmp::metal::Texture;

} // namespace metal

namespace oc {

using hmp::oc::PixelBuffer;

} // namespace oc

} // namespace bmf_sdk

BMF_DEFINE_TYPE(bmf_sdk::metal::Texture);
BMF_DEFINE_TYPE(bmf_sdk::metal::Device);
BMF_DEFINE_TYPE(bmf_sdk::oc::PixelBuffer);