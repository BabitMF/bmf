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
#include <sstream>
#include <hmp/tensor.h>
#include <hmp/core/stream.h>
#include <kernel/tensor_factory.h>
#include <hmp/format.h>

namespace hmp{

//////
std::string stringfy(const Tensor &tensor)
{
    if(tensor.defined()){
        Tensor tmp = tensor;
        //
        if(!tmp.is_contiguous()){
            tmp = tmp.contiguous();
        }

        if(!tmp.is_cpu()){
            auto t = empty_like(tmp, tmp.options().device(kCPU));
            copy(t, tmp);
            tmp = t;
        }

        std::stringstream ss;
        ss << fmt::format("Tensor({}, {}, {})\n", 
            tensor.device(),
            tensor.scalar_type(),
            tensor.shape());

        if(tmp.dim() == 1){
            tmp = tmp.reshape({1, 1, tmp.size(0)});
        }
        else if(tmp.dim() == 2){
            tmp = tmp.reshape({1, tmp.size(0), tmp.size(1)});
        }
        else{
            tmp = tmp.reshape({-1, tmp.size(-2), tmp.size(-1)});
        }

        HMP_DISPATCH_ALL_TYPES_AND_HALF(tmp.scalar_type(), "stringfy", [&](){
            auto oriShape = tensor.shape();
            auto shape = tmp.shape();

            for(int64_t i = 0; i < shape[0]; ++i){
                if(oriShape.size() <= 2){
                    //do nothing
                }
                else{

                }

                if(i != 0){
                    ss << ",\n";
                }

                auto tab = " ";
                if(tensor.dim() >= 2){
                    ss << "[";
                }
                for(int64_t j = 0; j < shape[1]; ++j){
                    auto ptr = tmp.select(0, i).select(0, j).data<scalar_t>();

                    if(j != 0){
                        ss << ",\n";
                    }

                    if(j != 0){
                        ss << tab << "[";
                    }
                    else{
                        ss << "[";
                    }
                    for(int64_t k = 0; k < shape[2]; ++k){
                        if(k != 0){
                            ss << ", ";
                        }

                        ss << fmt::format("{}", ptr[k]);
                    }
                    ss << "]";

                }

                if(tensor.dim() >= 2){
                    ss << "]";
                }
            }
        });

        
        return ss.str();
    }
    else{
        return "Tensor(Undefined)";
    }
}


} //namespace