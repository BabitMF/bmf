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

#import "MlStableDiffusionOC.h"
#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

@implementation MlStableDiffusionOC {
}

- (void)loadAndInit {
  self.excutor = [[SDExcutor alloc] init];
  [self.excutor loadAndInit];
}

- (double)getProgressValue {
  return [self.excutor getProgressValue];
}

- (NSString *)getPreparationPhase {
  return [self.excutor getPreparationPhase];
}

- (bool)hasCompleted {
  return [self.excutor hasCompleted];
}

- (int)getStatus {
  return [self.excutor getStatus];
}

- (CGImageRef)getResult {
  bool completed = [self.excutor hasCompleted];
  if (completed) {
    return [self.excutor getResult];
  }
  return nil;
}

- (void)generateImageWithPrompt:(NSString *)text
                       WithStep:(double)steps
                        AndSeed:(uint32_t)seeds {
  [self.excutor generateImageWithPrompt:text steps:steps seed:seeds];
}

@end
