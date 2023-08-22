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
#import <bmf/oc/Packet.h>

@interface BmfTask : NSObject

- (id)initFromPtr:(void *)task own:(bool)own;
- (id)init:(int)node_id
    istream_ids:(NSMutableArray *)istream_ids
    ostream_ids:(NSMutableArray *)ostream_ids;
- (void)dealloc;

- (void *)ptr;
- (void)setTimestamp:(long)ts;
- (long)timestamp;
- (bool)fillInputPacket:(int)stream_id pkt:(BmfPacket *)pkt;
- (bool)fillOutputPacket:(int)stream_id pkt:(BmfPacket *)pkt;
- (BmfPacket *)popPacketFromOutQueue:(int)stream_id;
- (BmfPacket *)popPacketFromInQueue:(int)stream_id;
- (NSMutableArray *)getInputStreamIds;
- (NSMutableArray *)getOutputStreamIds;

@end
