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
#import "test_ModuleFunctor.h"
#import <hmp/oc/Device.h>
#import "hmp/oc/Formats.h"
#import <hmp/oc/Device.h>
#import <hmp/oc/Scalar.h>
#import <hmp/oc/Image.h>
#import <hmp/oc/ScalarType.h>
#import <hmp/oc/Stream.h>
#import <hmp/oc/Tensor.h>
#import "bmf/oc/OpaqueDataKey.h"
#import <bmf/oc/Rational.h>
#import "bmf/oc/ModuleFunctor.h"

#define IMPORT_MODULE_CLASS(module_name)\
    extern "C" void* Constructor_##module_name##ModuleTag; \
    static void* _Constructor_##module_name##ModuleTag = Constructor_##module_name##ModuleTag;

IMPORT_MODULE_CLASS(VeryfastDenoiseModule)


@implementation BmfModuleFunctorTests

- (int) testAll
{
    char* ch = "{\"name\":\"ios_passthrough_module\",\"option\":{\"path\":\"my_path\",\"entry\":\"my_entry\"}}";
    NSString *string = [[NSString alloc] initWithCString:ch encoding:NSUTF8StringEncoding];
    NSError *error = nil;
    NSData *data= [string dataUsingEncoding:NSUTF8StringEncoding];
    id json_object = [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingAllowFragments error:&error];
    BmfModuleFunctor *mf = [[BmfModuleFunctor alloc]init:"VeryfastDenoiseModule" type:"c++" path:"" entry:"" option:json_object ninputs:2 noutputs:2];
    if ([mf isKindOfClass:[BmfModuleFunctor class]] != 1) {
        return 1;
    }

    return 0;
}

@end
