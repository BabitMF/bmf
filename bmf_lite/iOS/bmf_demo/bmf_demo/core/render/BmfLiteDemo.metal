#include <metal_stdlib>
using namespace metal;

kernel void psnr_kernel(texture2d<float, access::read> inputTexture1 [[texture(0)]],
                        texture2d<float, access::read> inputTexture2 [[texture(1)]],
                        device float* psnr [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    float4 pixel1 = inputTexture1.read(gid);
    float4 pixel2 = inputTexture2.read(gid);
    float4 diff = pixel1 - pixel2;
    // Mean Squared Error
    float mse = dot(diff, diff) / 4.0f;
    float maxPixelValue = 1.0f;
    // PSNR
    float psnrValue = 20.0f * log10(maxPixelValue) - 10.0f * log10(mse);
    psnr[0] = psnrValue;
}

#define YUV_SHADER_ARGS  VertexOut      inFrag    [[ stage_in ]],\
texture2d<float>  yTex     [[ texture(0) ]],\
texture2d<float>  uTex     [[ texture(1) ]],\
texture2d<float>  vTex     [[ texture(2) ]],\
sampler bilinear [[ sampler(0) ]], \
 constant ColorParameters *colorParameters [[ buffer(0) ]]

#define NV12_SHADER_ARGS  VertexOut      inFrag    [[ stage_in ]],\
texture2d<float>  yTex     [[ texture(0) ]],\
texture2d<float>  uvTex     [[ texture(1) ]],\
sampler bilinear [[ sampler(0) ]], \
constant ColorParameters *colorParameters [[ buffer(0) ]]

#define RGB_SHADER_ARGS  VertexOut      inFrag    [[ stage_in ]],\
texture2d<float>  tex     [[ texture(0) ]],\
sampler bilinear [[ sampler(0) ]]

#define Y_SHADER_ARGS  \
texture2d<float, access::read>  sourceTexture   [[texture(0)]], \
device          float          *buffer         [[buffer(0)]], \
uint2                          grid            [[thread_position_in_grid]]

struct VertexIn{
    packed_float3 position;
    packed_float2 st;
};

struct VertexOut{
    float4 position [[position]];  //1
    float2 st;
};

struct ColorParameters
{
    float3x3 yuvToRGB;
    float3 offset;
    float3 rangeMin;
    float3 rangeMax;
    float line;
};

struct ColorConversion {
    float3x3 matrix;
    float3 offset;
    float3 rangeMin;
    float3 rangeMax;
    float line;
};

vertex VertexOut texture_vertex(
                              const device VertexIn* vertex_array [[ buffer(0) ]],           //1
                              unsigned int vid [[ vertex_id ]]) {
    
    
    VertexIn VertexIn = vertex_array[vid];
    
    VertexOut VertexOut;
    VertexOut.position = float4(VertexIn.position,1);  //3
    VertexOut.st = VertexIn.st;
    return VertexOut;
}

fragment float4 texture_fragment(VertexOut frag [[stage_in]], texture2d<float> texas[[texture(0)]]) {  //1
    constexpr sampler defaultSampler;
    float4 rgba = texas.sample(defaultSampler, frag.st).rgba;
    return rgba;
}

fragment half4 yuv_rgb(YUV_SHADER_ARGS)
{
    float3 yuv;
    yuv.x = yTex.sample(bilinear, inFrag.st).r - float(16.0/255.0);
    yuv.y = uTex.sample(bilinear,inFrag.st).r - float(0.5);
    yuv.z = vTex.sample(bilinear,inFrag.st).r - float(0.5);
    return half4(half3(colorParameters->yuvToRGB * yuv),1.0);
}

//fragment half4 nv12_rgb(NV12_SHADER_ARGS)
//{
//    float3 yuv;
//    yuv.x = yTex.sample(bilinear, inFrag.st).r - float(16.0/255.0);
//    yuv.yz = uvTex.sample(bilinear,inFrag.st).rg - float2(0.5);
//    return half4(half3(colorParameters->yuvToRGB * yuv),1.0);
//}

fragment half4 nv12_rgb(NV12_SHADER_ARGS)
{
    float3 yuv;
    yuv.x = yTex.sample(bilinear, inFrag.st).r - colorParameters->offset[0];
    yuv.yz = uvTex.sample(bilinear,inFrag.st).rg - colorParameters->offset[1];

    //    return half4(clamp(half3(colorParameters->yuvToRGB * yuv), half3(16.0/1023, 16.0/1023, 16.0/1023), half3(2.0, 2.0, 2.0)),1.0);
    return half4(half3(colorParameters->yuvToRGB * yuv), 1.0);
}

fragment half4 nv12_rgb_compare_rotate(VertexOut inFrag [[ stage_in ]],
                                texture2d<float> yTex [[ texture(0) ]],
                                texture2d<float> uvTex [[ texture(1) ]],
                                texture2d<float> c_yTex [[ texture(2) ]],
                                texture2d<float> c_uvTex [[ texture(3) ]],
                                sampler bilinear [[ sampler(0) ]],
                                constant ColorParameters *colorParameters [[ buffer(0) ]])
{
    float3 yuv;
    float line = colorParameters->line;
    float low = saturate(line - 0.001f);
    float hi =  saturate(line + 0.001f);
    if (inFrag.st.g < low) {
        yuv.x = yTex.sample(bilinear, inFrag.st).r - colorParameters->offset[0];
        yuv.yz = uvTex.sample(bilinear,inFrag.st).rg - colorParameters->offset[1];
        return half4(half3(colorParameters->yuvToRGB * yuv), 1.0);
    } else if (low <= inFrag.st.g && inFrag.st.g <= hi) {
        return half4(1.0, 0.0, 0.0, 0.0);
    } else {
        yuv.x = c_yTex.sample(bilinear, inFrag.st).r - colorParameters->offset[0];
        yuv.yz = c_uvTex.sample(bilinear,inFrag.st).rg - colorParameters->offset[1];
        return half4(half3(colorParameters->yuvToRGB * yuv), 1.0);
    }    
}

fragment half4 nv12_rgb_compare(VertexOut inFrag [[ stage_in ]],
                                texture2d<float> yTex [[ texture(0) ]],
                                texture2d<float> uvTex [[ texture(1) ]],
                                texture2d<float> c_yTex [[ texture(2) ]],
                                texture2d<float> c_uvTex [[ texture(3) ]],
                                sampler bilinear [[ sampler(0) ]],
                                constant ColorParameters *colorParameters [[ buffer(0) ]])
{
    float3 yuv;
    float line = colorParameters->line;
    float low = saturate(line - 0.001f);
    float hi =  saturate(line + 0.001f);
    if (inFrag.st.r < low) {
        yuv.x = yTex.sample(bilinear, inFrag.st).r - colorParameters->offset[0];
        yuv.yz = uvTex.sample(bilinear,inFrag.st).rg - colorParameters->offset[1];
        return half4(half3(colorParameters->yuvToRGB * yuv), 1.0);
    } else if (low <= inFrag.st.r && inFrag.st.r <= hi) {
        return half4(1.0, 0.0, 0.0, 0.0);
    } else {
        yuv.x = c_yTex.sample(bilinear, inFrag.st).r - colorParameters->offset[0];
        yuv.yz = c_uvTex.sample(bilinear,inFrag.st).rg - colorParameters->offset[1];
        return half4(half3(colorParameters->yuvToRGB * yuv), 1.0);
    }

//        float3 yuv;
//        float line = colorParameters->line;
//        float low = saturate(line - 0.001f);
//        float hi =  saturate(line + 0.001f);
//        if (inFrag.st.r < low) {
//            yuv.x = yTex.sample(bilinear, inFrag.st).r - colorParameters->offset[0];
//            yuv.yz = uvTex.sample(bilinear,inFrag.st).rg - colorParameters->offset[1];
//            return half4(half3(colorParameters->yuvToRGB * yuv), 1.0);
//        } else if (low <= inFrag.st.r && inFrag.st.r <= hi) {
//            return half4(1.0, 0.0, 0.0, 0.0);
//        } else {
//            yuv.x = c_yTex.sample(bilinear, float2(inFrag.st.r - line, inFrag.st.g)).r - colorParameters->offset[0];
//            yuv.yz = c_uvTex.sample(bilinear, float2(inFrag.st.r - line, inFrag.st.g)).rg - colorParameters->offset[1];
//            return half4(half3(colorParameters->yuvToRGB * yuv), 1.0);
//        }
    
}


fragment half4 rgb_render(RGB_SHADER_ARGS)
{
    return half4(tex.sample(bilinear, inFrag.st).rgba);
}

kernel void y_tofloatbuffer(Y_SHADER_ARGS)
{
    if(grid.x >= sourceTexture.get_width() || grid.y >= sourceTexture.get_height()) return;

    float y  = sourceTexture.read(grid).r;
    buffer[(grid.y*sourceTexture.get_width() + grid.x)] = y;
}

kernel void pro_rgb_nv12_compute_ypass(texture2d<float, access::write>  yTexture  [[texture(0)]],
                                texture2d<float, access::read> rgbTexture [[texture(1)]],
                                uint2                          gid        [[thread_position_in_grid]],
                                constant ColorConversion& colorConversion [[ buffer(0) ]])
{
     if((gid.x >= rgbTexture.get_width()) || (gid.y >= rgbTexture.get_height()))
    {
        return;
    }
    float3 yuv = colorConversion.matrix * rgbTexture.read(gid).rgb + colorConversion.offset;
    yuv = clamp(yuv, colorConversion.rangeMin, colorConversion.rangeMax);
    yTexture.write(float4(yuv.xyzz), gid);
}

kernel void pro_nv12_rgb_compute_ypass(texture2d<float, access::read>  yTexture   [[texture(0)]],
                                       texture2d<float, access::read>  uvTexture  [[texture(1)]],
                                       texture2d<float, access::write> rgbTexture [[texture(2)]],
                                       uint2              gid        [[thread_position_in_grid]],
                                       constant ColorConversion& colorConversion [[ buffer(0) ]])
{
     if((gid.x >= uvTexture.get_width()) || (gid.y >= uvTexture.get_height()))
    {
        return;
    }
    uint2 gid_y = gid * 2;
    float y00 = yTexture.read(uint2(gid_y.x + 0, gid_y.y + 0)).x - float(colorConversion.offset.x);
    float y01 = yTexture.read(uint2(gid_y.x + 1, gid_y.y + 0)).x - float(colorConversion.offset.x);
    float y10 = yTexture.read(uint2(gid_y.x + 0, gid_y.y + 1)).x - float(colorConversion.offset.x);
    float y11 = yTexture.read(uint2(gid_y.x + 1, gid_y.y + 1)).x - float(colorConversion.offset.x);
    float2 uv = uvTexture.read(gid).xy - float2(0.5);
    float3 yuv00 = float3(y00, uv);
    float3 yuv01 = float3(y01, uv);;
    float3 yuv10 = float3(y10, uv);;
    float3 yuv11 = float3(y11, uv);;
    yuv00 = colorConversion.matrix * yuv00;
    yuv01 = colorConversion.matrix * yuv01;
    yuv10 = colorConversion.matrix * yuv10;
    yuv11 = colorConversion.matrix * yuv11;
    yuv00 = clamp(yuv00, colorConversion.rangeMin, colorConversion.rangeMax);
    yuv01 = clamp(yuv01, colorConversion.rangeMin, colorConversion.rangeMax);
    yuv10 = clamp(yuv10, colorConversion.rangeMin, colorConversion.rangeMax);
    yuv11 = clamp(yuv11, colorConversion.rangeMin, colorConversion.rangeMax);
    rgbTexture.write(float4(yuv00, 1.0), uint2(gid_y.x + 0, gid_y.y + 0));
    rgbTexture.write(float4(yuv01, 1.0), uint2(gid_y.x + 1, gid_y.y + 0));
    rgbTexture.write(float4(yuv10, 1.0), uint2(gid_y.x + 0, gid_y.y + 1));
    rgbTexture.write(float4(yuv11, 1.0), uint2(gid_y.x + 1, gid_y.y + 1));
}

kernel void pro_rgb_nv12_compute_uvpass(texture2d<float, access::write> uvTexture [[texture(0)]],
                                texture2d<float, access::read> rgbTexture [[texture(1)]],
                                uint2                          gid        [[thread_position_in_grid]],
                                constant ColorConversion& colorConversion [[ buffer(0) ]])
{
     if((gid.x >= rgbTexture.get_width()) || (gid.y >= rgbTexture.get_height()))
    {
        return;
    }
    float3 yuv = colorConversion.matrix * rgbTexture.read(gid).rgb + colorConversion.offset;
    yuv = clamp(yuv, colorConversion.rangeMin, colorConversion.rangeMax);
    if (gid.x % 2 == 0 && gid.y % 2 == 0) {
        uvTexture.write(float4(yuv.yzzz), uint2(gid.x/2, gid.y/2));
    }
}
