#include <metal_stdlib>
using namespace metal;

#define TEMPORAL_SLOPE 15.0h

inline half get_luma(half4 rgba) {
    return dot(half3(0.299h, 0.587h, 0.114h), rgba.rgb);
}

inline half gaussian(half x, half s, half m) {
    half scaled = (x - m) / s;
    return exp(-0.5h * scaled * scaled);
}

kernel void denoise( 
            texture2d<half, access::read> in_rgba [[texture(0)]],
            texture2d<half, access::read> pre_rgba_tex [[texture(1)]],
            texture2d<half, access::write> out_rgba [[texture(2)]],
            texture2d<half, access::write> cur_rgba_tex [[texture(3)]],
             uint2 gid    [[thread_position_in_grid]],
             uint2 lid [[thread_position_in_threadgroup]])
{
    int2 pos = int2(gid);
    int width = in_rgba.get_width();
    int height = in_rgba.get_height();
    
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    half4 histogram_v[5][5];
    half n = 0.0h;
    half4 sum = half4(0.0);

    half is = 0.05h;
    half ss = 1.0h;

    int x_m_1 = max(pos.x - 1, 0);
    int x_a_1 = min(pos.x + 1, width - 1);
    int x_m_2 = max(pos.x - 2, 0);
    int x_a_2 = min(pos.x + 2, width - 1);

    int y_m_1 = max(pos.y - 1, 0);
    int y_a_1 = min(pos.y + 1, height - 1);
    int y_m_2 = max(pos.y - 2, 0);
    int y_a_2 = min(pos.y + 2, height - 1);

    histogram_v[0][0] = in_rgba.read(uint2(x_m_2, y_m_2));
    histogram_v[1][0] = in_rgba.read(uint2(x_m_1, y_m_2));
    histogram_v[2][0] = in_rgba.read(uint2(pos.x, y_m_2));
    histogram_v[3][0] = in_rgba.read(uint2(x_a_1, y_m_2));
    histogram_v[4][0] = in_rgba.read(uint2(x_a_2, y_m_2));

    histogram_v[0][1] = in_rgba.read(uint2(x_m_2, y_m_1));
    histogram_v[1][1] = in_rgba.read(uint2(x_m_1, y_m_1));
    histogram_v[2][1] = in_rgba.read(uint2(pos.x, y_m_1));
    histogram_v[3][1] = in_rgba.read(uint2(x_a_1, y_m_1));
    histogram_v[4][1] = in_rgba.read(uint2(x_a_2, y_m_1));

    histogram_v[0][2] = in_rgba.read(uint2(x_m_2, pos.y));
    histogram_v[1][2] = in_rgba.read(uint2(x_m_1, pos.y));
    histogram_v[2][2] = in_rgba.read(uint2(pos.x, pos.y));
    histogram_v[3][2] = in_rgba.read(uint2(x_a_1, pos.y));
    histogram_v[4][2] = in_rgba.read(uint2(x_a_2, pos.y));

    histogram_v[0][3] = in_rgba.read(uint2(x_m_2, y_a_1));
    histogram_v[1][3] = in_rgba.read(uint2(x_m_1, y_a_1));
    histogram_v[2][3] = in_rgba.read(uint2(pos.x, y_a_1));
    histogram_v[3][3] = in_rgba.read(uint2(x_a_1, y_a_1));
    histogram_v[4][3] = in_rgba.read(uint2(x_a_2, y_a_1));

    histogram_v[0][4] = in_rgba.read(uint2(x_m_2, y_a_2));
    histogram_v[1][4] = in_rgba.read(uint2(x_m_1, y_a_2));
    histogram_v[2][4] = in_rgba.read(uint2(pos.x, y_a_2));
    histogram_v[3][4] = in_rgba.read(uint2(x_a_1, y_a_2));
    histogram_v[4][4] = in_rgba.read(uint2(x_a_2, y_a_2));

    half vc = get_luma(histogram_v[2][2]);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            half w = gaussian(get_luma(histogram_v[i][j]), is, vc) * gaussian(length(half2(i - 2, j - 2)), ss, 0.0h);
            n += w;
            sum += histogram_v[i][j] * w;
        }
    }

    half4 result = sum / n;

    half result_y = get_luma(result);
    half4 pre_rgba = pre_rgba_tex.read(gid);
    half pre_y = get_luma(pre_rgba);
    half temporal_weight = max(min(abs(result_y - pre_y) * TEMPORAL_SLOPE, 1.0h), 0.0h);
    result = mix((pre_rgba + result) * 0.5h, result, temporal_weight);

    out_rgba.write(result, gid);
    cur_rgba_tex.write(result, gid);
    return;
}


kernel void denoise_no_mix( 
            texture2d<half, access::read> in_rgba [[texture(0)]],
            texture2d<half, access::write> out_rgba [[texture(2)]],
            texture2d<half, access::write> cur_rgba_tex [[texture(3)]],
             uint2 gid    [[thread_position_in_grid]],
             uint2 lid [[thread_position_in_threadgroup]])
{
    int2 pos = int2(gid);
    int width = in_rgba.get_width();
    int height = in_rgba.get_height();
    
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    half4 histogram_v[5][5];
    half n = 0.0h;
    half4 sum = half4(0.0);
    
    half is = 0.05h;
    half ss = 1.0h;

    int x_m_1 = max(pos.x - 1, 0);
    int x_a_1 = min(pos.x + 1, width - 1);
    int x_m_2 = max(pos.x - 2, 0);
    int x_a_2 = min(pos.x + 2, width - 1);

    int y_m_1 = max(pos.y - 1, 0);
    int y_a_1 = min(pos.y + 1, height - 1);
    int y_m_2 = max(pos.y - 2, 0);
    int y_a_2 = min(pos.y + 2, height - 1);

    histogram_v[0][0] = in_rgba.read(uint2(x_m_2, y_m_2));
    histogram_v[1][0] = in_rgba.read(uint2(x_m_1, y_m_2));
    histogram_v[2][0] = in_rgba.read(uint2(pos.x, y_m_2));
    histogram_v[3][0] = in_rgba.read(uint2(x_a_1, y_m_2));
    histogram_v[4][0] = in_rgba.read(uint2(x_a_2, y_m_2));

    histogram_v[0][1] = in_rgba.read(uint2(x_m_2, y_m_1));
    histogram_v[1][1] = in_rgba.read(uint2(x_m_1, y_m_1));
    histogram_v[2][1] = in_rgba.read(uint2(pos.x, y_m_1));
    histogram_v[3][1] = in_rgba.read(uint2(x_a_1, y_m_1));
    histogram_v[4][1] = in_rgba.read(uint2(x_a_2, y_m_1));

    histogram_v[0][2] = in_rgba.read(uint2(x_m_2, pos.y));
    histogram_v[1][2] = in_rgba.read(uint2(x_m_1, pos.y));
    histogram_v[2][2] = in_rgba.read(uint2(pos.x, pos.y));
    histogram_v[3][2] = in_rgba.read(uint2(x_a_1, pos.y));
    histogram_v[4][2] = in_rgba.read(uint2(x_a_2, pos.y));

    histogram_v[0][3] = in_rgba.read(uint2(x_m_2, y_a_1));
    histogram_v[1][3] = in_rgba.read(uint2(x_m_1, y_a_1));
    histogram_v[2][3] = in_rgba.read(uint2(pos.x, y_a_1));
    histogram_v[3][3] = in_rgba.read(uint2(x_a_1, y_a_1));
    histogram_v[4][3] = in_rgba.read(uint2(x_a_2, y_a_1));

    histogram_v[0][4] = in_rgba.read(uint2(x_m_2, y_a_2));
    histogram_v[1][4] = in_rgba.read(uint2(x_m_1, y_a_2));
    histogram_v[2][4] = in_rgba.read(uint2(pos.x, y_a_2));
    histogram_v[3][4] = in_rgba.read(uint2(x_a_1, y_a_2));
    histogram_v[4][4] = in_rgba.read(uint2(x_a_2, y_a_2));

    half vc = get_luma(histogram_v[2][2]);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            half w = gaussian(get_luma(histogram_v[i][j]), is, vc) * gaussian(length(half2(i - 2, j - 2)), ss, 0.0h);
            n += w;
            sum += histogram_v[i][j] * w;
        }
    }

    half4 result = sum / n;

    out_rgba.write(result, gid);
    cur_rgba_tex.write(result, gid);
    return;
}

kernel void denoise_nv12( 
            texture2d<half, access::read> in_y [[texture(0)]],
            texture2d<half, access::read> pre_y_tex [[texture(1)]],
            texture2d<half, access::write> out_y [[texture(2)]],
            texture2d<half, access::write> cur_y_tex [[texture(3)]],
            texture2d<half, access::read> in_uv [[texture(4)]],
            texture2d<half, access::write> out_uv [[texture(5)]],
             uint2 gid    [[thread_position_in_grid]],
             uint2 lid [[thread_position_in_threadgroup]])
{
    int2 pos = int2(gid);
    pos = pos << 1;
    int width = in_y.get_width();
    int height = in_y.get_height();
    
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    half histogram_v[6][6];
    half n00 = 0.0h;
    half n01 = 0.0h;
    half n10 = 0.0h;
    half n11 = 0.0h;
    half sum00 = half(0.0);
    half sum01 = half(0.0);
    half sum10 = half(0.0);
    half sum11 = half(0.0);

    half is = 0.05h;
    half ss = 1.0h;

    int x_m_1 = max(pos.x - 1, 0);
    int x_a_1 = pos.x + 1;
    int x_m_2 = max(pos.x - 2, 0);
    int x_a_2 = min(pos.x + 2, width - 1);
    int x_a_3 = min(pos.x + 3, width - 1);

    int y_m_1 = max(pos.y - 1, 0);
    int y_a_1 = pos.y + 1;
    int y_m_2 = max(pos.y - 2, 0);
    int y_a_2 = min(pos.y + 2, height - 1);
    int y_a_3 = min(pos.y + 3, height - 1);

    histogram_v[0][0] = in_y.read(uint2(x_m_2, y_m_2)).r;
    histogram_v[1][0] = in_y.read(uint2(x_m_1, y_m_2)).r;
    histogram_v[2][0] = in_y.read(uint2(pos.x, y_m_2)).r;
    histogram_v[3][0] = in_y.read(uint2(x_a_1, y_m_2)).r;
    histogram_v[4][0] = in_y.read(uint2(x_a_2, y_m_2)).r;
    histogram_v[5][0] = in_y.read(uint2(x_a_3, y_m_2)).r;

    histogram_v[0][1] = in_y.read(uint2(x_m_2, y_m_1)).r;
    histogram_v[1][1] = in_y.read(uint2(x_m_1, y_m_1)).r;
    histogram_v[2][1] = in_y.read(uint2(pos.x, y_m_1)).r;
    histogram_v[3][1] = in_y.read(uint2(x_a_1, y_m_1)).r;
    histogram_v[4][1] = in_y.read(uint2(x_a_2, y_m_1)).r;
    histogram_v[5][1] = in_y.read(uint2(x_a_3, y_m_1)).r;

    histogram_v[0][2] = in_y.read(uint2(x_m_2, pos.y)).r;
    histogram_v[1][2] = in_y.read(uint2(x_m_1, pos.y)).r;
    histogram_v[2][2] = in_y.read(uint2(pos.x, pos.y)).r;
    histogram_v[3][2] = in_y.read(uint2(x_a_1, pos.y)).r;
    histogram_v[4][2] = in_y.read(uint2(x_a_2, pos.y)).r;
    histogram_v[5][2] = in_y.read(uint2(x_a_3, pos.y)).r;

    histogram_v[0][3] = in_y.read(uint2(x_m_2, y_a_1)).r;
    histogram_v[1][3] = in_y.read(uint2(x_m_1, y_a_1)).r;
    histogram_v[2][3] = in_y.read(uint2(pos.x, y_a_1)).r;
    histogram_v[3][3] = in_y.read(uint2(x_a_1, y_a_1)).r;
    histogram_v[4][3] = in_y.read(uint2(x_a_2, y_a_1)).r;
    histogram_v[5][3] = in_y.read(uint2(x_a_3, y_a_1)).r;

    histogram_v[0][4] = in_y.read(uint2(x_m_2, y_a_2)).r;
    histogram_v[1][4] = in_y.read(uint2(x_m_1, y_a_2)).r;
    histogram_v[2][4] = in_y.read(uint2(pos.x, y_a_2)).r;
    histogram_v[3][4] = in_y.read(uint2(x_a_1, y_a_2)).r;
    histogram_v[4][4] = in_y.read(uint2(x_a_2, y_a_2)).r;
    histogram_v[5][4] = in_y.read(uint2(x_a_3, y_a_2)).r;

    histogram_v[0][5] = in_y.read(uint2(x_m_2, y_a_3)).r;
    histogram_v[1][5] = in_y.read(uint2(x_m_1, y_a_3)).r;
    histogram_v[2][5] = in_y.read(uint2(pos.x, y_a_3)).r;
    histogram_v[3][5] = in_y.read(uint2(x_a_1, y_a_3)).r;
    histogram_v[4][5] = in_y.read(uint2(x_a_2, y_a_3)).r;
    histogram_v[5][5] = in_y.read(uint2(x_a_3, y_a_3)).r;

    half vc00 = histogram_v[2][2];
    half vc01 = histogram_v[3][2];
    half vc10 = histogram_v[2][3];
    half vc11 = histogram_v[3][3];

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            half w = gaussian(length(half2(i - 2, j - 2)), ss, 0.0h);
            half w00 = gaussian(histogram_v[i][j], is, vc00) * w;
            n00 += w00;
            sum00 += histogram_v[i][j] * w00;
            half w01 = gaussian(histogram_v[i + 1][j], is, vc01) * w;
            n01 += w01;
            sum01 += histogram_v[i + 1][j] * w01;
            half w10 = gaussian(histogram_v[i][j + 1], is, vc10) * w;
            n10 += w10;
            sum10 += histogram_v[i][j + 1] * w10;
            half w11 = gaussian(histogram_v[i + 1][j + 1], is, vc11) * w;
            n11 += w11;
            sum11 += histogram_v[i + 1][j + 1] * w11;
        }
    }

    sum00 = sum00 / n00;
    sum01 = sum01 / n01;
    sum10 = sum10 / n10;
    sum11 = sum11 / n11;

    half pre_y00 = pre_y_tex.read(uint2(pos.x, pos.y)).r;
    half pre_y01 = pre_y_tex.read(uint2(x_a_1, pos.y)).r;
    half pre_y10 = pre_y_tex.read(uint2(pos.x, y_a_1)).r;
    half pre_y11 = pre_y_tex.read(uint2(x_a_1, y_a_1)).r;

    half temporal_weight00 = max(min(abs(sum00 - pre_y00) * TEMPORAL_SLOPE, 1.0h), 0.0h);
    half temporal_weight01 = max(min(abs(sum01 - pre_y01) * TEMPORAL_SLOPE, 1.0h), 0.0h);
    half temporal_weight10 = max(min(abs(sum10 - pre_y10) * TEMPORAL_SLOPE, 1.0h), 0.0h);
    half temporal_weight11 = max(min(abs(sum11 - pre_y11) * TEMPORAL_SLOPE, 1.0h), 0.0h);

    sum00 = mix((pre_y00 + sum00) * 0.5h, sum00, temporal_weight00);
    sum01 = mix((pre_y01 + sum01) * 0.5h, sum01, temporal_weight01);
    sum10 = mix((pre_y10 + sum10) * 0.5h, sum10, temporal_weight10);
    sum11 = mix((pre_y11 + sum11) * 0.5h, sum11, temporal_weight11);

    out_y.write(sum00, uint2(pos.x, pos.y));
    out_y.write(sum01, uint2(x_a_1, pos.y));
    out_y.write(sum10, uint2(pos.x, y_a_1));
    out_y.write(sum11, uint2(x_a_1, y_a_1));

    cur_y_tex.write(sum00, uint2(pos.x, pos.y));
    cur_y_tex.write(sum01, uint2(x_a_1, pos.y));
    cur_y_tex.write(sum10, uint2(pos.x, y_a_1));
    cur_y_tex.write(sum11, uint2(x_a_1, y_a_1));

    out_uv.write(in_uv.read(gid), gid);

    return;
}

kernel void denoise_nv12_no_mix(
            texture2d<half, access::read> in_y [[texture(0)]],
            texture2d<half, access::write> out_y [[texture(2)]],
            texture2d<half, access::write> cur_y_tex [[texture(3)]],
            texture2d<half, access::read> in_uv [[texture(4)]],
            texture2d<half, access::write> out_uv [[texture(5)]],
            uint2 gid    [[thread_position_in_grid]],
            uint2 lid [[thread_position_in_threadgroup]])
{
    int2 pos = int2(gid);
    pos = pos << 1;
    int width = in_y.get_width();
    int height = in_y.get_height();
    
    if (pos.x >= width || pos.y >= height) {
        return;
    }

    half histogram_v[6][6];
    half n00 = 0.0h;
    half n01 = 0.0h;
    half n10 = 0.0h;
    half n11 = 0.0h;
    half sum00 = half(0.0);
    half sum01 = half(0.0);
    half sum10 = half(0.0);
    half sum11 = half(0.0);

    half is = 0.05h;
    half ss = 1.0h;

    int x_m_1 = max(pos.x - 1, 0);
    int x_a_1 = pos.x + 1;
    int x_m_2 = max(pos.x - 2, 0);
    int x_a_2 = min(pos.x + 2, width - 1);
    int x_a_3 = min(pos.x + 3, width - 1);

    int y_m_1 = max(pos.y - 1, 0);
    int y_a_1 = pos.y + 1;
    int y_m_2 = max(pos.y - 2, 0);
    int y_a_2 = min(pos.y + 2, height - 1);
    int y_a_3 = min(pos.y + 3, height - 1);

    histogram_v[0][0] = in_y.read(uint2(x_m_2, y_m_2)).r;
    histogram_v[1][0] = in_y.read(uint2(x_m_1, y_m_2)).r;
    histogram_v[2][0] = in_y.read(uint2(pos.x, y_m_2)).r;
    histogram_v[3][0] = in_y.read(uint2(x_a_1, y_m_2)).r;
    histogram_v[4][0] = in_y.read(uint2(x_a_2, y_m_2)).r;
    histogram_v[5][0] = in_y.read(uint2(x_a_3, y_m_2)).r;

    histogram_v[0][1] = in_y.read(uint2(x_m_2, y_m_1)).r;
    histogram_v[1][1] = in_y.read(uint2(x_m_1, y_m_1)).r;
    histogram_v[2][1] = in_y.read(uint2(pos.x, y_m_1)).r;
    histogram_v[3][1] = in_y.read(uint2(x_a_1, y_m_1)).r;
    histogram_v[4][1] = in_y.read(uint2(x_a_2, y_m_1)).r;
    histogram_v[5][1] = in_y.read(uint2(x_a_3, y_m_1)).r;

    histogram_v[0][2] = in_y.read(uint2(x_m_2, pos.y)).r;
    histogram_v[1][2] = in_y.read(uint2(x_m_1, pos.y)).r;
    histogram_v[2][2] = in_y.read(uint2(pos.x, pos.y)).r;
    histogram_v[3][2] = in_y.read(uint2(x_a_1, pos.y)).r;
    histogram_v[4][2] = in_y.read(uint2(x_a_2, pos.y)).r;
    histogram_v[5][2] = in_y.read(uint2(x_a_3, pos.y)).r;

    histogram_v[0][3] = in_y.read(uint2(x_m_2, y_a_1)).r;
    histogram_v[1][3] = in_y.read(uint2(x_m_1, y_a_1)).r;
    histogram_v[2][3] = in_y.read(uint2(pos.x, y_a_1)).r;
    histogram_v[3][3] = in_y.read(uint2(x_a_1, y_a_1)).r;
    histogram_v[4][3] = in_y.read(uint2(x_a_2, y_a_1)).r;
    histogram_v[5][3] = in_y.read(uint2(x_a_3, y_a_1)).r;

    histogram_v[0][4] = in_y.read(uint2(x_m_2, y_a_2)).r;
    histogram_v[1][4] = in_y.read(uint2(x_m_1, y_a_2)).r;
    histogram_v[2][4] = in_y.read(uint2(pos.x, y_a_2)).r;
    histogram_v[3][4] = in_y.read(uint2(x_a_1, y_a_2)).r;
    histogram_v[4][4] = in_y.read(uint2(x_a_2, y_a_2)).r;
    histogram_v[5][4] = in_y.read(uint2(x_a_3, y_a_2)).r;

    histogram_v[0][5] = in_y.read(uint2(x_m_2, y_a_3)).r;
    histogram_v[1][5] = in_y.read(uint2(x_m_1, y_a_3)).r;
    histogram_v[2][5] = in_y.read(uint2(pos.x, y_a_3)).r;
    histogram_v[3][5] = in_y.read(uint2(x_a_1, y_a_3)).r;
    histogram_v[4][5] = in_y.read(uint2(x_a_2, y_a_3)).r;
    histogram_v[5][5] = in_y.read(uint2(x_a_3, y_a_3)).r;

    half vc00 = histogram_v[2][2];
    half vc01 = histogram_v[3][2];
    half vc10 = histogram_v[2][3];
    half vc11 = histogram_v[3][3];

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            half w = gaussian(length(half2(i - 2, j - 2)), ss, 0.0h);
            half w00 = gaussian(histogram_v[i][j], is, vc00) * w;
            n00 += w00;
            sum00 += histogram_v[i][j] * w00;
            half w01 = gaussian(histogram_v[i + 1][j], is, vc01) * w;
            n01 += w01;
            sum01 += histogram_v[i + 1][j] * w01;
            half w10 = gaussian(histogram_v[i][j + 1], is, vc10) * w;
            n10 += w10;
            sum10 += histogram_v[i][j + 1] * w10;
            half w11 = gaussian(histogram_v[i + 1][j + 1], is, vc11) * w;
            n11 += w11;
            sum11 += histogram_v[i + 1][j + 1] * w11;
        }
    }

    sum00 = sum00 / n00;
    sum01 = sum01 / n01;
    sum10 = sum10 / n10;
    sum11 = sum11 / n11;

    out_y.write(sum00, uint2(pos.x, pos.y));
    out_y.write(sum01, uint2(x_a_1, pos.y));
    out_y.write(sum10, uint2(pos.x, y_a_1));
    out_y.write(sum11, uint2(x_a_1, y_a_1));

    cur_y_tex.write(sum00, uint2(pos.x, pos.y));
    cur_y_tex.write(sum01, uint2(x_a_1, pos.y));
    cur_y_tex.write(sum10, uint2(pos.x, y_a_1));
    cur_y_tex.write(sum11, uint2(x_a_1, y_a_1));

    out_uv.write(in_uv.read(gid), gid);

    return;
}