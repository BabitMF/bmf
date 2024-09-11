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

#ifndef _BMFLITE_DEMO_VOD_PLAYER_H_
#define _BMFLITE_DEMO_VOD_PLAYER_H_
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>

@interface BmfLiteDemoVodPlayer : NSObject

- (instancetype)initWithMTKView:(MTKView *)view
                   AndVideoPath:(NSString *)path
               WhetherPlayAudio:(BOOL)play_audio
                        Compare:(BOOL)compare_source;

- (void)setSliderValue:(float)value;

- (void)run;

- (void)stop;

@end

#endif /* _BMFLITE_DEMO_VOD_PLAYER_H_ */
