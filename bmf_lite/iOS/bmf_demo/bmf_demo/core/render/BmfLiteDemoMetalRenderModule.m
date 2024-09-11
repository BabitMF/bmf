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

#import "BmfLiteDemoMetalRenderModule.h"
#import "BmfLiteDemoErrorCode.h"
#import "BmfLiteDemoLog.h"
#import <CoreAudio/CoreAudioTypes.h>

BMFLITE_DEMO_NAMESPACE_BEGIN

MetalRenderModule::MetalRenderModule(MTKView* view, int render_mode, int rotate) {
    view_ = view;
    rotate_ = rotate;
    dispatch_async(dispatch_get_main_queue(), ^{
        layer_ = (CAMetalLayer*) view_.layer;
    });
    switch (render_mode) {
        case RenderMode::NV12:
            mode_ = RenderMode::NV12;
            break;
        case RenderMode::YUV420:
            mode_ = RenderMode::YUV420;
            break;
        case RenderMode::RGBA:
            mode_ = RenderMode::RGBA;
            break;
        default:
            mode_ = RenderMode::NV12;
            break;
    }
}

int MetalRenderModule::init() {
    render_ = [[BmfLiteViewRender alloc] initWithMetalKitView : view_ WhetherRotate:rotate_];
    [render_ mtkView:view_ drawableSizeWillChange:view_.drawableSize];
    view_.delegate = render_;
    return BmfLiteErrorCode::SUCCESS;
}

MetalRenderModule::~MetalRenderModule() {

}

void MetalRenderModule::checkAndUpdataLayerInfo(CVPixelBufferRef ibuffer) {
    if (ibuffer == nil) {
        return;
    }

    CFTypeRef color_attachments = CVBufferGetAttachment(ibuffer, kCVImageBufferColorPrimariesKey, NULL);
    CFTypeRef color_trans = CVBufferGetAttachment(ibuffer, kCVImageBufferTransferFunctionKey, NULL);
    CFStringRef color_space = kCGColorSpaceSRGB;
    BOOL notSDR = true;
    if (@available(iOS 11.0, *)) {
        if ([(__bridge NSString *)color_attachments isEqualToString:(__bridge NSString *)kCVImageBufferColorPrimaries_ITU_R_2020] && [(__bridge NSString *)color_trans isEqualToString:(__bridge NSString *)kCVImageBufferTransferFunction_ITU_R_2100_HLG]) {
            if (@available(iOS 14.0, *)) {
                color_space = kCGColorSpaceITUR_2100_HLG;
            } else {
                // Fallback on earlier versions
            }
        } else if ([(__bridge NSString *)color_attachments isEqualToString:(__bridge NSString *)kCVImageBufferColorPrimaries_ITU_R_2020] && [(__bridge NSString *)color_trans isEqualToString:(__bridge NSString *)kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ]) {
            if (@available(iOS 14.0, *)) {
                color_space = kCGColorSpaceITUR_2100_PQ;
            } else {
                // Fallback on earlier versions
            }
        } else if ([(__bridge NSString *)color_attachments isEqualToString:(__bridge NSString *)kCVImageBufferColorPrimaries_P3_D65] && [(__bridge NSString *)color_trans isEqualToString:(__bridge NSString *)kCVImageBufferTransferFunction_ITU_R_2100_HLG]) {
            if (@available(iOS 12.6, *)) {
                color_space = kCGColorSpaceDisplayP3_HLG;
            } else {
                // Fallback on earlier versions
            }
        } else if ([(__bridge NSString *)color_attachments isEqualToString:(__bridge NSString *)kCVImageBufferColorPrimaries_P3_D65] && [(__bridge NSString *)color_trans isEqualToString:(__bridge NSString *)kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ]) {
            if (@available(iOS 13.4, *)) {
                color_space = kCGColorSpaceDisplayP3_PQ;
            } else {
                // Fallback on earlier versions
            }
        } else if ([(__bridge NSString *)color_attachments isEqualToString:(__bridge NSString *)kCVImageBufferColorPrimaries_P3_D65]) {
            color_space = kCGColorSpaceDisplayP3;
        } else {
            notSDR = false;
        }
    } else {
        notSDR = false;
        // Fallback on earlier versions
    }
    
    if (![(__bridge NSString *)color_space isEqualToString:(__bridge NSString *)color_space_]) {
        if ([[NSThread currentThread] isMainThread]) {
            [CATransaction begin];
            [CATransaction setDisableActions:YES];
            color_space_ = color_space;
            layer_.colorspace = CGColorSpaceCreateWithName(color_space_);
            MTLPixelFormat fmt = view_.colorPixelFormat;
            if (notSDR) {
                view_.colorPixelFormat = MTLPixelFormatRGBA16Float;
            } else {
                view_.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
            }

            [CATransaction commit];
        } else {
                [CATransaction begin];
                [CATransaction setDisableActions:YES];
                color_space_ = color_space;
                if (layer_) {
                    layer_.colorspace = CGColorSpaceCreateWithName(color_space_);
                    if (notSDR) {
                        view_.colorPixelFormat = MTLPixelFormatRGBA16Float;
                    } else {
                        view_.colorPixelFormat = MTLPixelFormatBGRA8Unorm;
                    }
                }
                [CATransaction commit];
        }
    }
    
    if (@available(iOS 16.0, *)) {
        if ([(__bridge NSString *)color_trans isEqualToString:(__bridge NSString *)kCVImageBufferTransferFunction_ITU_R_2100_HLG]) {
                [CATransaction begin];
                [CATransaction setDisableActions:YES];
                if (layer_) {
                    [layer_ setWantsExtendedDynamicRangeContent:YES];
                }
                [CATransaction commit];
            
            CFTypeRef ambientData = CVBufferGetAttachment(ibuffer, kCVImageBufferAmbientViewingEnvironmentKey, NULL);
            if (CAEDRMetadata.available) {
                if (ambientData) {
//                    if (@available(iOS 17.2, *)) {
//                        layer_.EDRMetadata = [CAEDRMetadata HLGMetadataWithAmbientViewingEnvironment:((__bridge NSData *)ambientData)];
//                    } else {
                        layer_.EDRMetadata = CAEDRMetadata.HLGMetadata;
//                    }
                } else {
                    layer_.EDRMetadata = CAEDRMetadata.HLGMetadata;
//                    layer_.EDRMetadata = [CAEDRMetadata HDR10MetadataWithMinLuminance:0.1 maxLuminance:1000 opticalOutputScale:10000];
                    
                }
            }
        } else if ( [(__bridge NSString *)color_trans isEqualToString:(__bridge NSString *)kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ]) {
            dispatch_async(dispatch_get_main_queue(), ^{
                [CATransaction begin];
                [CATransaction setDisableActions:YES];
                if (layer_) {
                    [layer_ setWantsExtendedDynamicRangeContent:YES];
                }
                [CATransaction commit];
            });
            if (CAEDRMetadata.available) {
                layer_.EDRMetadata = [CAEDRMetadata HDR10MetadataWithMinLuminance:0.1 maxLuminance:1000 opticalOutputScale:10000];
            }
        } else {
                [CATransaction begin];
                [CATransaction setDisableActions:YES];
                if (layer_) {
                    [layer_ setWantsExtendedDynamicRangeContent:NO];
                }
                [CATransaction commit];
            layer_.EDRMetadata = nil;
        }
    } else {
        // Fallback on earlier versions
    }
}

void MetalRenderModule::setSliderValue(float value) {
    [render_ setSliderValue:value];
}

int MetalRenderModule::process(std::shared_ptr<VideoFrame> data) {

    if (nullptr == data) {
        return BmfLiteErrorCode::VIDEO_FRAME_IS_NIL;
    }
    if (data->eos_) {
        return BmfLiteErrorCode::SUCCESS;
    }
//    dispatch_async(dispatch_get_main_queue(), ^{
        checkAndUpdataLayerInfo(data->buffer_);
//    });
    OSType fmt;
    size_t w, h;
    if (data->buffer_) {
        fmt = CVPixelBufferGetPixelFormatType(data->buffer_);
        w = CVPixelBufferGetWidth(data->buffer_);
        h = CVPixelBufferGetHeight(data->buffer_);
    } else {
        fmt = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
        w = data->tex0_.width;
        h = data->tex0_.height;
    }

    if (data->first_ || pre_fmt_ != fmt) {
        size_t view_w = view_.drawableSize.width;
        size_t view_h = view_.drawableSize.height;
        [render_ setRenderPipelineConfig:fmt :(int)w :(int)h :(int)view_w :(int)view_h : (bool) data->compare_ : (float)0.5f];
        pre_fmt_ = fmt;
    }
    if (nil == view_) return BmfLiteErrorCode::SURFACE_IS_NIL;

    if (data->buffer_ != nil && data->tex0_ == nil) {
        auto& helper = MetalHelper::getSingleInstance();
        helper.createMTLTextureByCVPixelBufferRef(data->buffer_, &(data->tex0_), &(data->tex1_), &(data->tex2_));
    }

    if (data->compare_) {
        if (data->source_ != nil) {
            auto& helper = MetalHelper::getSingleInstance();
            helper.createMTLTextureByCVPixelBufferRef(data->source_, &(data->s_tex0_), &(data->s_tex1_), &(data->s_tex2_));
        }
    }

    if (data->compare_) {
        [render_ setMTLTexture:data->s_tex0_ :data->s_tex1_ : data->s_tex2_ : data->tex0_ :data->tex1_ : data->tex2_ : data->source_];
    } else {
        [render_ setMTLTexture:data->tex0_ :data->tex1_ : data->tex2_ : data->s_tex0_ :data->s_tex1_ : data->s_tex2_ : nil];
    }


    @autoreleasepool {
        Float64 p = CMTimeGetSeconds(data->p_time_) / CMTimeGetSeconds(data->duration_);
        NSNumber *tmp = [NSNumber numberWithDouble:p];
    }
    return BmfLiteErrorCode::SUCCESS;
}

int MetalRenderModule::close() {
    return BmfLiteErrorCode::SUCCESS;
}

BMFLITE_DEMO_NAMESPACE_END
