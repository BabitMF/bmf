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
#import <Foundation/Foundation.h>
#import <bmf/oc/VideoFrame.h>
#import <bmf/oc/AudioFrame.h>

@interface BmfPacket : NSObject

- (id)initFromPtr:(void *)pkt own:(bool)own;
- (id)init:(id)data;
- (void)dealloc;

+ (BmfPacket *)generateEosPacket;
+ (BmfPacket *)generateEofPacket;

- (void *)ptr;
- (bool)defined;
- (void)setTimestamp:(long)ts;
- (long)timestamp;
- (bool)is:(Class)class_type;
- (id)get:(Class)class_type;

@end
