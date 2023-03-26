#import <hmp/oc/Image.h>
#import <test_Image.h>

@implementation BmfImageTests

- (int) testAll
{
    //Image constructors
    @autoreleasepool{
        HmpImage *img0 = [[HmpImage alloc] init : 1920 : 1080 : 3 : kNHWC : kUInt8];
        if(![img0 defined]){
            return 1;
        }

        if([img0 format] != kNHWC){
            return 2;
        }

        HmpColorModel *cm = [img0 color_model];
        if([cm space] != CS_UNSPECIFIED){
            return 3;
        }

        if([img0 hdim] != 0 || [img0 wdim] != 1 || [img0 cdim] != 2){
            return 4;
        }

        if([img0 width] != 1920 || [img0 height] != 1080 || [img0 nchannels] != 3){
            return 5;
        }

        if([img0 dtype] != kUInt8){
            return 6;
        }

        if([[img0 device] type] != kCPU){
            return 7;
        }

        HmpTensor *data0 = [img0 data];
        if(![data0 defined] || [data0 unsafe_data] != [img0 unsafe_data]){
            return 8;
        }

        //
        HmpImage *img1 = [[HmpImage alloc] init : data0 : kNHWC];
        if([img1 width] != 1920 || [img1 height] != 1080 || [img1 nchannels] != 3){
            return 9;
        }

        HmpImage *img2 = [img0 clone];
        if([img2 width] != 1920 || [img2 height] != 1080 || [img2 nchannels] != 3){
            return 10;
        }

        [img2 copy_ : img0]; //only check api exists

        //
        HmpImage *img3 = [img0 crop : 100 : 200 : 200 : 300];
        if([img3 width] != 200 || [img3 height] != 300 || [img3 nchannels] != 3){
            return 11;
        }

        HmpImage *img4 = [img0 select : 1];
        if([img4 width] != 1920 || [img4 height] != 1080 || [img4 nchannels] != 1){
            return 12;
        }
    } //


    @autoreleasepool{
        HmpColorModel *bt709 = [[HmpColorModel alloc] init : CS_BT709 : CR_MPEG];
        HmpPixelInfo *yuv420p = [[HmpPixelInfo alloc] initEx : PF_YUV420P : bt709];
        HmpFrame *frame = [[HmpFrame alloc] init : 1920 : 1080 : yuv420p];

        if(![frame defined]){
            return 20;
        }

        HmpPixelInfo *pix_info = [frame pix_info];
        if([pix_info format] != PF_YUV420P || [pix_info range] != CR_MPEG 
            || [pix_info space] != CS_BT709 || [pix_info primaries] != CP_UNSPECIFIED
            || [pix_info transfer_characteristic] != CTC_UNSPECIFIED){
            return 21;
        }

        if([frame format] != PF_YUV420P || [frame width] != 1920 
            || [frame height] != 1080 || [frame nplanes] != 3){
            return 22;
        }

        if([frame dtype] != kUInt8){
            return 23;
        }

        if([[frame device] type] != kCPU){
            return 24;
        }

        //
        HmpTensor *Y = [frame plane : 0];
        if([Y size : 0] != 1080 || [Y size : 1] != 1920){
            return 25;
        }

        HmpTensor *U = [frame plane : 1];
        if([U size : 0] != 540 || [U size : 1] != 960){
            return 26;
        }

        HmpTensor *V = [frame plane : 2];
        if([V size : 0] != 540 || [V size : 1] != 960){
            return 27;
        }

        //
        if([frame plane_data : 0] != [Y unsafe_data] 
            || [frame plane_data : 1] != [U unsafe_data] 
            || [frame plane_data : 2] != [V unsafe_data]){
            return 28;
        }

        HmpFrame *frame1 = [[HmpFrame alloc] init : 1920 : 1080 : yuv420p];
        [frame1 copy_ : frame]; //only check api exists

        HmpFrame *frame2 = [frame crop : 100 : 100 : 200 : 300];
        if([frame2 format] != PF_YUV420P || [frame2 width] != 200
            || [frame2 height] != 300 || [frame2 nplanes] != 3){
            return 29;
        }

        HmpPixelInfo *rgb24 = [[HmpPixelInfo alloc] initEx : PF_RGB24 : bt709];
        HmpFrame *frame3 = [[HmpFrame alloc] init : 1920 : 1080 : rgb24];

        HmpImage *img3 = [frame3 to_image : kNHWC];
        if([img3 width] != 1920 || [img3 height] != 1080 || [img3 nchannels] != 3){
            return 30;
        }

        HmpFrame *frame4 = [HmpFrame from_image : img3 : rgb24];
        if([frame4 format] != PF_RGB24 || [frame4 width] != 1920 
            || [frame4 height] != 1080 || [frame4 nplanes] != 1){
            return 31;
        }
    }

    // init from pixel buffer(NV12)
    @autoreleasepool{
        NSDictionary *pixelAttributes = @{(NSString*)kCVPixelBufferIOSurfacePropertiesKey:@{}};
        CVPixelBufferRef pixelBuffer = NULL;
        CVReturn result = CVPixelBufferCreate(kCFAllocatorDefault,
                                          1920,
                                          1080,
                                          kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                                          (__bridge CFDictionaryRef)(pixelAttributes),
                                          &pixelBuffer);//kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        if(result != kCVReturnSuccess){
            NSLog(@"Unable to create cvpixelbuffer %d", result);
            return 60;
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, 0);
        HmpFrame *frame0 = [[HmpFrame alloc] initFromPixelBuffer : pixelBuffer];
        if(!frame0 || [frame0 width] != 1920 || [frame0 height] != 1080 || [frame0 nplanes] != 2){
            return 61;
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

        HmpTensor *Y = [frame0 plane : 0];
        HmpTensor *UV = [frame0 plane : 1];
        if([Y size : 0] != 1080 || [Y size : 1] != 1920 || 
           [UV size : 0] != 540 || [UV size : 1] != 960 || [UV size : 2] != 2){
            return 62;
        }

        if([[frame0 pix_info] range] != CR_JPEG){
            return 63;
        }
        
        CVPixelBufferRelease(pixelBuffer);

    }

    // init from pixel buffer(YUV420P)
    @autoreleasepool{
        NSDictionary *pixelAttributes = @{(NSString*)kCVPixelBufferIOSurfacePropertiesKey:@{}};
        CVPixelBufferRef pixelBuffer = NULL;
        CVReturn result = CVPixelBufferCreate(kCFAllocatorDefault,
                                          1920,
                                          1080,
                                          kCVPixelFormatType_420YpCbCr8Planar,
                                          (__bridge CFDictionaryRef)(pixelAttributes),
                                          &pixelBuffer);//kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        if(result != kCVReturnSuccess){
            NSLog(@"Unable to create cvpixelbuffer %d", result);
            return 70;
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, 0);
        HmpFrame *frame0 = [[HmpFrame alloc] initFromPixelBuffer : pixelBuffer];
        if(!frame0 || [frame0 width] != 1920 || [frame0 height] != 1080 || [frame0 nplanes] != 3){
            return 71;
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

        HmpTensor *Y = [frame0 plane : 0];
        HmpTensor *U = [frame0 plane : 1];
        HmpTensor *V = [frame0 plane : 1];
        if([Y size : 0] != 1080 || [Y size : 1] != 1920 || 
           [U size : 0] != 540 || [U size : 1] != 960 || [U size : 2] != 1 ||
           [V size : 0] != 540 || [V size : 1] != 960 || [V size : 2] != 1
           ){
            return 72;
        }
        
        CVPixelBufferRelease(pixelBuffer);
    }


    return 0;
}

@end
