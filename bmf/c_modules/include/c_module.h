/*
 * Copyright 2023 Babit Authors
 *
 * This file is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 */

#ifndef BMF_C_MODULE_H
#define BMF_C_MODULE_H

#include <cstdlib>
#include <sstream>
#include <bmf/sdk/log.h>
#include <bmf/sdk/module.h>
#include <bmf/sdk/module_registry.h>
#include <bmf/sdk/task.h>
#include <bmf/sdk/packet.h>
#include <bmf/sdk/json_param.h>

enum {
    PROCESS_OK = 0,
    PROCESS_STOP,
    PROCESS_ERROR
};

enum {
    VIDEO_TYPE = 1,
    PICTURELIST_TYPE = 2,
    VIDEOLIST_TYPE = 3
};

USE_BMF_SDK_NS

#endif //BMF_C_MODULE_H
