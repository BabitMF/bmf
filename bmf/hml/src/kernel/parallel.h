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

#include <hmp/tensor.h>

#ifdef _OPENMP
#define INTRA_OP_PARALLEL
#include <omp.h>
#endif

namespace hmp{
namespace kernel{

template<typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t step, const F &f)
{
    HMP_REQUIRE(step >= 0, "parallel_for: invalid step {}", step);
    if(begin >= end){
        return;
    }

#ifdef _OPENMP
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;
    //
    int64_t num_threads = omp_in_parallel() ? 1 : omp_get_max_threads();
    if(step > 0){
        num_threads = std::min(num_threads, divup((end - begin), step));
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int64_t num_threads = omp_get_num_threads();
        int64_t tid = omp_get_thread_num();
        int64_t chunk_size = divup((end - begin), num_threads);
        int64_t begin_tid = begin + tid * chunk_size;
        if(begin_tid < end){
            try{
                f(begin_tid, std::min(end, begin_tid + chunk_size));
            }
            catch(...){
                if(!err_flag.test_and_set()){
                    eptr = std::current_exception();
                }
            }
        }
    }

    if(eptr){
        std::rethrow_exception(eptr);
    }

#else
    f(begin, end);
#endif




}


}} // hmp::kernel