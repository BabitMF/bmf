#include <bmf/sdk/bmf_type_info.h>
#include <hmp/oc/CV.h>
#include <hmp/oc/Metal.h>

namespace bmf_sdk{
namespace metal{

using hmp::metal::Texture;
using hmp::metal::Device;

} //namespace metal

namespace oc{

using hmp::oc::PixelBuffer;

} //namespace oc

} //namespace bmf_sdk


BMF_DEFINE_TYPE(bmf_sdk::metal::Texture);
BMF_DEFINE_TYPE(bmf_sdk::metal::Device);
BMF_DEFINE_TYPE(bmf_sdk::oc::PixelBuffer);