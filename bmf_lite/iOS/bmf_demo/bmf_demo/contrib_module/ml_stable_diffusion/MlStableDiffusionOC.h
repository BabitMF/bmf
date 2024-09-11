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

#ifndef MlStableDiffusionOC_h
#define MlStableDiffusionOC_h

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>
#import "bmf_demo-Swift.h"

@interface MlStableDiffusionOC : NSObject

@property(nonatomic, retain) SDExcutor *excutor;

- (void)loadAndInit;

- (double)getProgressValue;

- (NSString *)getPreparationPhase;

- (bool)hasCompleted;

- (CGImageRef)getResult;

- (int)getStatus;

- (void)generateImageWithPrompt:(NSString *)text
                       WithStep:(double)steps
                        AndSeed:(uint32_t)seeds;

@end

#endif /* MlStableDiffusionOC_h */
