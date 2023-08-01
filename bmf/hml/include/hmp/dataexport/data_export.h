#pragma once

#include <hmp/tensor.h>
#include <dlpack/dlpack.h>

namespace hmp {
HMP_API DLManagedTensor *to_dlpack(const Tensor &src);
HMP_API Tensor from_dlpack(const DLManagedTensor *src);
} // namespace hmp