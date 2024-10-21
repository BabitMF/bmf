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
#import <test_Device.h>
#import <test_Scalar.h>
#import <test_Tensor.h>
#import <test_Image.h>
#import <test_CV.h>
#import <test_Metal.h>

static void test_main()
{
    int total = 0;
    int pass = 0;

    int result = 0;

    result = [[BmfDeviceTests alloc] testAll];
    NSLog(@"BmfDeviceTests    -> %d\n", result);
    pass += result == 0;
    total += 1;


    result = [[BmfScalarTests alloc] testAll];
    NSLog(@"BmfScalarTests    -> %d\n", result);
    pass += result == 0;
    total += 1;


    result = [[BmfTensorTests alloc] testAll];
    NSLog(@"BmfTensorTests    -> %d\n", result);
    pass += result == 0;
    total += 1;


    result = [[BmfImageTests alloc] testAll];
    NSLog(@"BmfImageTests    -> %d\n", result);
    pass += result == 0;
    total += 1;

    result = [[BmfCVTests alloc] testAll];
    NSLog(@"BmfCVTests    -> %d\n", result);
    pass += result == 0;
    total += 1;

    result = [[BmfMetalTests alloc] testAll];
    NSLog(@"BmfMetalTests    -> %d\n", result);
    pass += result == 0;
    total += 1;

    NSLog(@"Total:%d, Pass:%d, Fail:%d\n", total, pass, total - pass);
    return 0;
}

int main(int argc, const char * argv[]) 
{
    for(int i = 0; i < 10000; ++i){ //NOTE: loop to test memleaks
        NSLog(@"=================== %03d ======================\n", i);
        test_main();
        NSLog(@"\n\n");
    }
    
    return 0;
}
