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
#import <bmf/oc/Rational.h>

@interface BmfRational()
@property (nonatomic, assign) int num;
@property (nonatomic, assign) int den;
@end

@implementation BmfRational

- (id)init: (int)num den:(int)den{
    if((self = [super init]) != nil){
        self.num = num;
        self.den = den;
    }
    return self;
}

- (int)num{
    return _num;
}

- (int)den{
    return _den;
}

@end
