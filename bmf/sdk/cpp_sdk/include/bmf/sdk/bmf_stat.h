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

#include <bmf/sdk/json_param.h>
#include <bmf/sdk/shared_library.h>

BEGIN_BMF_SDK_NS

class BMFStat
{
    static BMFStat& GetInstance();
    void push_info(JsonParam info);
    int upload_info(); //if the lib was exists, transfer the stat info back
    void dump(std::string filename); //dump to file or print directly

  private:
    BMFStat(); //single instance
    ~BMFStat() {};
    BMFStat(const BMFStat &bmfst) = delete;
    const BMFStat &operator= (const BMFStat &bmfst) = delete;
    JsonParam stat_info;
    SharedLibrary upload_lib;
};

END_BMF_SDK_NS
