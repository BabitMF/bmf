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
#import "hmp/imgproc/formats.h"
#import "hmp/oc/Formats.h"

using namespace hmp;


@interface HmpColorModel()
@property (nonatomic, assign) ColorModel* impl;
@property (nonatomic, assign) bool own;
@end


@implementation HmpColorModel

- (instancetype) init : (HmpColorSpace) cs : (HmpColorRange) cr
{
	self = [super init];
    if(self){
        self.impl = new ColorModel((ColorSpace)cs, (ColorRange)cr);
        self.own = true;
    }
    return self;
}

- (instancetype) initEx : (HmpColorSpace) cs : (HmpColorRange) cr : (HmpColorPrimaries) cp : (HmpColorTransferCharacteristic) ctc
{
	self = [super init];
    if(self){
        self.impl = new ColorModel((ColorSpace)cs, (ColorRange)cr, (ColorPrimaries)cp,
                                    (ColorTransferCharacteristic)ctc);
        self.own = true;
    }
    return self;
}

- (instancetype) initFromPtr: (void*) ptr : (bool) own
{
    self = [super init];
    self.impl = (ColorModel*)ptr;
    self.own = own;
    return self;
}

- (void*)ptr
{
    return self.impl;
}

- (void) dealloc
{
    if(self.own && self.impl){
        delete self.impl;
    }
}


- (HmpColorRange) range
{
    return (HmpColorRange)self.impl->range();
}

- (HmpColorSpace) space
{
    return (HmpColorSpace)self.impl->space();
}

- (HmpColorPrimaries) primaries
{
    return (HmpColorPrimaries)self.impl->primaries();
}

- (HmpColorTransferCharacteristic) transfer_characteristic
{
    return (HmpColorTransferCharacteristic)self.impl->transfer_characteristic();
}



@end //HmpColorModel


@interface HmpPixelInfo()
@property (nonatomic, assign) PixelInfo* impl;
@property (nonatomic, assign) bool own;
@end


@implementation HmpPixelInfo


- (instancetype) init : (HmpPixelFormat) format
{
    self = [super init];
    if(self){
        self.impl = new PixelInfo((PixelFormat)format);
        self.own = true;
    }
    return self;
}

- (instancetype) initEx : (HmpPixelFormat) format : (HmpColorModel*) model
{
    self = [super init];
    if(self){
        self.impl = new PixelInfo((PixelFormat)format, *(ColorModel*)[model ptr]);
        self.own = true;
    }
    return self;
}

- (instancetype) initFromPtr: (void*) ptr : (bool) own
{
    self = [super init];
    self.impl = (PixelInfo*)ptr;
    self.own = own;
    return self;
}

- (void*)ptr
{
    return self.impl;
}

- (void) dealloc
{
    if(self.own && self.impl){
        delete self.impl;
    }
}


- (HmpPixelFormat) format
{
    return (HmpPixelFormat)self.impl->format();
}

- (HmpColorRange) range
{
    return (HmpColorRange)self.impl->range();
}

- (HmpColorSpace) space
{
    return (HmpColorSpace)self.impl->range();
}

- (HmpColorPrimaries) primaries
{
    return (HmpColorPrimaries)self.impl->primaries();
}

- (HmpColorTransferCharacteristic) transfer_characteristic
{
    return (HmpColorTransferCharacteristic)self.impl->transfer_characteristic();
}

- (HmpColorSpace) infer_space
{
    return (HmpColorSpace)self.impl->infer_space();
}

- (bool) is_rgbx
{
    return self.impl->is_rgbx();
}


@end //HmpPixelInfo
