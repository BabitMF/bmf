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
#import "bmf/oc/ModuleFunctor.h"
#import "bmf/sdk/packet.h"
#import "bmf/sdk/video_frame.h"
#import "bmf/sdk/task.h"
#import "bmf/sdk/module_functor.h"

using namespace bmf_sdk;
using namespace hmp;

@interface BmfModuleFunctor()
@property (nonatomic, assign) ModuleFunctor* p;
@property (nonatomic, assign) bool own;
@end

@implementation BmfModuleFunctor

- (id)initFromPtr: (void*)mf own:(bool)own{
    if((self = [super init]) != nil){
        self.p = (ModuleFunctor*)mf;
        self.own = own;
    }
    return self;
}

- (id)init: (char *)name type:(char *)type path:(char*)path entry:(char *)entry option:(id)option ninputs:(int)ninputs noutputs:(int)noutputs
{
    if([NSJSONSerialization isValidJSONObject:option]){
        NSError *error = nil;
        NSData *data_str = [NSJSONSerialization dataWithJSONObject:option options:NSJSONReadingAllowFragments error:&error];
        NSString *ns_string = [[NSString alloc]initWithData:data_str encoding:NSUTF8StringEncoding];
        const char *opt =[ns_string UTF8String];
        
        auto &M = ModuleManager::instance();
        ModuleInfo info(name, type, entry, path);
        auto factory = M.load_module(info);
        if(factory == nullptr){
            throw std::runtime_error("Load module " + info.module_name + " failed");
        }
        JsonParam json_option;
        json_option.parse(opt);
        auto m = factory->make(-1, json_option);
        self.p = new ModuleFunctor(m, ninputs, noutputs);
        self.own = true;
    }else{
        NSLog(@"private attach did not get an actual jsonparam.");
    }
    return self;
}

- (void) dealloc{
    delete self.p;
}

- (void*)ptr{
    return self.p;
}

@end

