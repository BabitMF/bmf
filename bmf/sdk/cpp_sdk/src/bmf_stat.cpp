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
#include <bmf/sdk/bmf_stat.h>
#include <bmf/sdk/log.h>

namespace bmf_sdk {

BMFStat& BMFStat::GetInstance()
{
    static BMFStat bmfst;
    return bmfst;
}

BMFStat::BMFStat()
{
    std::string path = ""; //linux, win, ios, android
    upload_lib = SharedLibrary(path, SharedLibrary::LAZY | SharedLibrary::GLOBAL);
    if (upload_lib.is_open())
        BMFLOG(BMF_INFO) << "BMF stat upload lib was found and loaded";
}

void BMFStat::push_info(JsonParam info)
{
    stat_info.merge_patch(info);
}

int BMFStat::upload_info()
{
    if (upload_lib.is_open()) {
        //send back the stat info base on the lib for different OS/environments
    }
    return 0;
}

} // namespace bmf_sdk
