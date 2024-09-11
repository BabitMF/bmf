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

#ifndef _BMF_LITE_PARAM_OC_H_
#define _BMF_LITE_PARAM_OC_H_

#import <Foundation/Foundation.h>

@interface BmfParamOC : NSObject {
}

- (instancetype)init;

- (BOOL)hasKey:(NSString *)key;

- (BOOL)setInt:(int32_t)value WithKey:(NSString *)key;

- (int32_t)getIntByKey:(NSString *)key;

- (BOOL)setInt64:(int64_t)value WithKey:(NSString *)key;

- (int64_t)getInt64ByKey:(NSString *)key;

- (BOOL)setFloatValue:(Float32)value WithKey:(NSString *)key;

- (Float32)getFloatByKey:(NSString *)key;

- (BOOL)setDouble:(double)value WithKey:(NSString *)key;

- (double)getDoubleByKey:(NSString *)key;

- (BOOL)setString:(NSString *)value WithKey:(NSString *)key;

- (NSString *)getStringByKey:(NSString *)key;

@end

#endif // _BMF_LITE_PARAM_OC_H_
