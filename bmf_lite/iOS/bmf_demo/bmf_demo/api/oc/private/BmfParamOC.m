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

#import "BmfParamOC.h"
#import <Foundation/Foundation.h>

#include <string>
#include <bmf_lite.h>

@implementation BmfParamOC {
    bmf_lite::Param* param;
}

- (instancetype)init {
    if (self = [super init]) {
        param = nullptr;
        param = new (std::nothrow) bmf_lite::Param();
        if (param == nullptr) {
            return nil;
        }
    }
    return self;
}

- (BOOL)hasKey:(NSString *)key {
    if (nil == key) {
        return NO;
    }
    std::string c_key = std::string([key UTF8String]);
    return param->has_key(c_key);
}

- (BOOL)setInt:(int32_t)value WithKey:(NSString *)key {
    if (key == nil) {
        return NO;
    }
    std::string c_key = std::string([key UTF8String]);
    return param->setInt(c_key, value);

}

- (int32_t)getIntByKey:(NSString *)key {
    if (key == nil) {
        return -1;
    }
    std::string c_key = std::string([key UTF8String]);
    int value = 0;
    bool ret = param->getInt(c_key, value);
    if (ret) {
        return value;
    }
    return -1;
}

- (BOOL)setInt64:(int64_t)value WithKey:(NSString *)key {
    if (key == nil) {
        return NO;
    }
    std::string c_key = std::string([key UTF8String]);
    return param->setLong(c_key, value);
}

- (int64_t)getInt64ByKey:(NSString *)key {
    if (key == nil) {
        return 0;
    }
    std::string c_key = std::string([key UTF8String]);
    int64_t value;
    param->getLong(c_key, value);
    return value;
}

- (BOOL)setFloatValue:(Float32)value WithKey:(NSString *)key {
    if (key == nil) {
        return NO;
    }
    std::string c_key = std::string([key UTF8String]);
    return param->setFloat(c_key, value);
}

- (Float32)getFloatByKey:(NSString *)key {
    if (key == nil) {
        return 0.f;
    }
    std::string c_key = std::string([key UTF8String]);
    float value = 0.f;
    param->getFloat(c_key, value);
    return value;
}

- (BOOL)setDouble:(double)value WithKey:(NSString *)key {
    if (key == nil) {
        return NO;
    }
    std::string c_key = std::string([key UTF8String]);
    return param->setDouble(c_key,value);
}

- (double)getDoubleByKey:(NSString *)key {
    if (key == nil) {
        return NO;
    }
    std::string c_key = std::string([key UTF8String]);
    double value = 0.;
    param->getDouble(c_key, value);
    return value;
}

- (BOOL)setString:(NSString *)value WithKey:(NSString *)key {
    if (key == nil) {
        return NO;
    }
    std::string c_value = "";
    std::string c_key = std::string([key UTF8String]);
    if (key != nil) {
        c_value = [value UTF8String];
    }
    return param->setString(c_key, c_value);
}

- (NSString*)getStringByKey:(NSString *)key {
    if (key == nil) {
        return nil;
    }
    std::string c_key = std::string([key UTF8String]);
    std::string c_value = "";
    param->getString(c_key, c_value);
    NSString *ns_str = nil;
    if (c_value != "") {
        ns_str = [[NSString alloc] initWithCString:c_value.c_str()];
    }
    return ns_str;
}

- (void)dealloc {
    if (NULL != param) {
        delete param;
    }
    param = NULL;
}

@end
