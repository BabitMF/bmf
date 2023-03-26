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
#include <thread>
#include <gtest/gtest.h>
#include <hmp/core/ref_ptr.h>

namespace hmp{

namespace {

struct DummyObject : public RefObject
{
    std::atomic<int> count = 0;
};


} //namespace



TEST(RefPtr, multi_thread_ops)
{
    const int nthreads = 16;
    const int nops = 100000;

    auto obj = makeRefPtr<DummyObject>();

    std::atomic<int> wait_count = 0;
    auto op_func = [&](){
        wait_count += 1;
        std::vector<RefPtr<DummyObject>> copy_objs;
        std::vector<RefPtr<DummyObject>> move_objs;

        //wait
        while(wait_count >= 0); //barrier

        //copy
        for(int i = 0; i < nops; ++i){
            copy_objs.push_back(obj);
        }

        //move
        for(auto &o : copy_objs){
            o->count += 1;
            move_objs.push_back(std::move(o));
        }
    };

    //
    std::vector<std::thread> threads;
    for(int i = 0; i < nthreads; ++i){
        threads.push_back(std::thread(op_func));
    }

    //wait all threads startup
    while(wait_count != nthreads); //barrier

    //
    wait_count.store(-1); //

    //join
    for(auto &t : threads){
        t.join();
    }

    EXPECT_EQ(1, obj.refcount());
    EXPECT_EQ(nthreads*nops, obj->count);
}



} //namespace