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
#import <hmp/oc/Scalar.h>
#import <test_Scalar.h>
#include <hmp/core/scalar.h>

@implementation BmfScalarTests

- (int) testAll
{
    //constructors
    @autoreleasepool{
	    HmpScalar *s0 = [[HmpScalar alloc] init];
        if(![s0 is_integral : false]){
            return 1;
        }

	    HmpScalar *s1 = [[HmpScalar alloc] initFromBool: true];
        if(![s1 is_boolean] || [s1 to_bool] != true){
            return 2;
        }

	    HmpScalar *s2 = [[HmpScalar alloc] initFromInt: 42];
        if(![s2 is_integral : false] || [s2 to_int] != 42){
            return 3;
        }

	    HmpScalar *s3 = [[HmpScalar alloc] initFromFloat: 42.42];
        if(![s3 is_floating_point] || [s3 to_float] != 42.42){
            return 4;
        }
    }

    //
    @autoreleasepool{
        hmp::Scalar *ptr = new hmp::Scalar(42);
        HmpScalar *s0 = [[HmpScalar alloc] initFromPtr: ptr : true];
        if(![s0 is_integral : false] || [s0 to_int] != 42){
            return 10;
        }

    }

    return 0;
}

@end
