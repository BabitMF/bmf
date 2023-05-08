# encoding: utf-8
"""
Created on 2020-11-5

@author: hejianqiang
"""

import os
import time
import logging
import copy
from uuid import uuid4
import numpy as np
import cv2
from edit_layout import create_layout_extra

MIN_WH = 8


# def process_logo_frame_thread(frame_list, idx_list, layout_param_list):
#     from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
#     executor = ThreadPoolExecutor(max_workers=2)
#     all_task = []
#     for i, layout_param in enumerate(layout_param_list):
#         _task = executor.submit(process_logo_frame, (frame_list[idx_list[i]], layout_param))
#         all_task.append(_task)
#     wait(all_task, return_when=ALL_COMPLETED)


def colors_to_rgb(color):
    # color: '#rrggbb' -> [r, g, b]
    if not color:
        rgb = [0, 0, 0]
        return rgb

    if type(color) in [list, tuple]:
        return list(color)[0:3]

    if color.startswith('#'):
        _r = 0x00
        _g = 0x00
        _b = 0x00
        if len(color) >= 7:
            _r = int(color[1:3], 16)
            _g = int(color[3:5], 16)
            _b = int(color[5:7], 16)
        rgb = [_r, _g, _b]
        return rgb


def create_frame(width, height, dim=3, color=None, store_frame=0):
    """
    color=[255,0,0]
    """
    # frame = np.ones(shape=(width, height, dim), dtype=np.uint8)*255
    color = colors_to_rgb(color)
    bgr_color = tuple(reversed(color))
    frame = np.full((height, width, dim), bgr_color, dtype=np.uint8)
    # cv2.imshow('title', frame)
    if store_frame:
        cv2.imwrite('create_frame.jpg', frame)
    return frame


def process_logo_frame(logo_frame, watermark_param):
    if not watermark_param:
        return

    pos = watermark_param.get('pos')
    resolution_fill = watermark_param.get('crop_mode') or watermark_param.get('resolution_fill', 'pad')

    rotate = watermark_param.get('rotate', 0)
    if rotate in [90, -90, 180, 270]:
        _k = (rotate % 360)/90
        logo_frame = np.rot90(logo_frame, k=_k)
        # logo_frame = cv2.rotate(logo_frame, cv2.ROTATE_90_CLOCKWISE if _k==1 else cv2.ROTATE_90_COUNTERCLOCKWISE)

    if resolution_fill in ['crop']:
        logo_frame_scale = inner_frame_crop(logo_frame, pos[2], pos[3])
    elif resolution_fill in ['scale']:
        logo_frame_scale = inner_frame_scale(logo_frame, pos[2], pos[3])
    else:
        # color = watermark_param.get('color', [0, 0, 0])
        # logo_frame_scale = inner_frame_pad(logo_frame, pos[2], pos[3], color=color)
        logo_simu_info = inner_frame_pad_simulate(logo_frame, pos[2], pos[3])
        logo_frame_scale = logo_simu_info.get('frame')
        new_pos = logo_simu_info.get('pos')
        new_pos[0] = pos[0] + new_pos[0]
        new_pos[1] = pos[1] + new_pos[1]
        watermark_param['pos'] = new_pos

    watermark_param['logo_file'] = logo_frame_scale
    return logo_frame_scale


# no used yet
def process_frame_add_logo(frame, watermark_param):
    if not watermark_param:
        return

    logo_frame = watermark_param.get('logo_file')
    pos = watermark_param.get('pos')
    x1 = pos[0]
    y1 = pos[1]
    x2 = x1 + pos[2]
    y2 = y1 + pos[3]

    frame[y1:y2, x1:x2] = logo_frame
    return frame


def inner_frame_add_logo(frame, logo_frame, watermark_param):
    """
    watermark_param = {'pos': [x, y, w, h], 'alpha': 0.5}
    # logo_frame_a = cv2.addWeighted(frame[y1:y2, x1:x2], alpha, logo_frame, 1-alpha, 0)
    """
    if not watermark_param:
        return

    process_logo_frame(logo_frame, watermark_param)

    logo_frame_scale = watermark_param.get('logo_file')
    pos = watermark_param.get('pos')
    x1 = pos[0]
    y1 = pos[1]
    x2 = x1 + pos[2]
    y2 = y1 + pos[3]

    # prefix_str = uuid4().hex
    # cv2.imwrite('test_frame_%s.jpg' % prefix_str, frame)
    # cv2.imwrite('test_logo_%s.jpg' % prefix_str, logo_frame)
    # cv2.imwrite('test_logo_scale_%s.jpg' % prefix_str, logo_frame_scale)
    # print("y1:y2, x1:x2 = %s:%s, %s:%s | frame.shape=%s, logo_frame.shape=%s" % (y1,y2, x1,x2, frame.shape, logo_frame_scale.shape))
    frame[y1:y2, x1:x2] = logo_frame_scale
    # cv2.imwrite('test_frame_logo_%s.jpg' % prefix_str, frame)
    return frame


def inner_frame_scale(frame, width, height):
    if width < MIN_WH or height < MIN_WH:
        return

    sh, sw, dim = frame.shape
    if sw == width and sh == height:
        return frame

    new_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)  # INTER_AREA  INTER_NEAREST  INTER_LINEAR  INTER_CUBIC
    return new_frame


def inner_frame_pad_simulate(frame, width, height):
    if width < MIN_WH or height < MIN_WH:
        return
    sh, sw, dim = frame.shape
    aspect_ratio = 1.0*width/height
    # print("[inner_frame_pad]sw=%s, sh=%s, dim=%s, aspect_ratio=%s" % (sw, sh, dim, aspect_ratio))

    if abs(1.0*sw/sh - aspect_ratio) < 0.001:
        new_w = width
        new_h = height
        pos = [0, 0, new_w, new_h]
    elif 1.0*sw/sh > aspect_ratio:
        new_w = width
        new_h = int(new_w *sh / sw)
        pos = [0, int((height-new_h)/2), new_w, new_h]
    else:
        new_h = height
        new_w = int(new_h*sw/sh)
        pos = [int((width-new_w)/2), 0, new_w, new_h]

    new_frame = inner_frame_scale(frame, new_w, new_h)
    return {'frame': new_frame, 'pos': pos}


def inner_frame_pad(frame, width, height, color):
    # image = cv2.copyMakeBorder( src, top, bottom, left, right, borderType)
    if width < MIN_WH or height < MIN_WH:
        return
    sh, sw, dim = frame.shape
    aspect_ratio = 1.0*width/height
    # print("[inner_frame_pad]sw=%s, sh=%s, dim=%s, aspect_ratio=%s" % (sw, sh, dim, aspect_ratio))

    if abs(1.0*sw/sh - aspect_ratio) < 0.001:
        # print("[inner_frame_pad]just scale")
        new_frame = inner_frame_scale(frame, width, height)
    else:
        _top, _bottom, _left, _right = 0, 0, 0, 0
        if 1.0*sw/sh > aspect_ratio:
            _w = width
            _h = int(_w*sh/sw)
            _top = int((height-_h)/2)
            _bottom = height-_h-_top

        else:
            _h = height
            _w = int(_h*sw/sh)
            _left = int((width-_w)/2)
            _right = width-_w-_left

        scale_frame = inner_frame_scale(frame, _w, _h)
        color = colors_to_rgb(color)
        bgr_color = list(reversed(color))
        new_frame = cv2.copyMakeBorder(scale_frame, _top, _bottom, _left, _right, cv2.BORDER_CONSTANT, value=bgr_color)  # cv2.BORDER_DEFAULT

        # full_frame = np.full((height, width, dim), color, dtype=np.uint8)
        # xx = int((width-_w)/2)
        # yy = int((height-_h)/2)
        # full_frame[yy:yy+_h, xx:xx+_w] = scale_frame
    return new_frame


def inner_frame_crop(frame, width, height):
    if width < MIN_WH or height < MIN_WH:
        return
    sh, sw, dim = frame.shape
    aspect_ratio = 1.0*width/height

    if abs(1.0*sw/sh - aspect_ratio) < 0.001:
        new_frame = inner_frame_scale(frame, width, height)
    else:
        if 1.0*sw/sh > aspect_ratio:
            _h = height
            _w = int(_h * sw / sh)
        else:
            _w = width
            _h = int(_w * sh / sw)

        scale_frame = inner_frame_scale(frame, _w, _h)
        sw, sh = _w, _h

        if 1.0 * sh / sw > 1.0 / aspect_ratio:
            new_h = int(sw * 1.0 / aspect_ratio)
            new_h += (new_h % 2)
            pos = [0, int((sh - new_h) / 2), sw, new_h]
        else:
            new_w = int(sh * aspect_ratio)
            new_w += (new_w % 2)
            pos = [int((sw - new_w) / 2), 0, new_w, sh]

        # roi = im[y1:y2, x1:x2] (x1,y1) top-left vertex and (x2,y2) as bottom-right vertex of a rectangle region within that image
        new_frame = scale_frame[pos[1]:pos[1]+pos[3], pos[0]:pos[0]+pos[2]]
    return new_frame


def get_layout_extra(frame_num, layout_option):
    width = layout_option.get('width', 1280)
    height = layout_option.get('height', 720)
    background_color = layout_option.get('background_color', "#000000")
    layout_mode = layout_option.get('layout_mode', 'speaker')
    layout_location = layout_option.get('layout_location', 'right')
    crop_mode = layout_option.get('crop_mode') or layout_option.get('resolution_fill') or 'pad'
    main_stream_crop_mode = layout_option.get('main_stream_crop_mode', '')
    interspace = layout_option.get('interspace', 0)

    rotate_list = layout_option.get('rotate_list')

    if layout_mode in ['custom'] and layout_option.get('layout_extra') and layout_option.get('layout_extra', {}).get('layout_param_list'):
        layout_extra = layout_option.get('layout_extra')
    else:
        layout_extra = create_layout_extra(frame_num=frame_num, width=width, height=height, color=background_color, resolution_fill=crop_mode,
                                       layout_location=layout_location, layout_mode=layout_mode)

    if layout_extra:
        layout_param_list = layout_extra.get('layout_param_list')
        for i, layout_param in enumerate(layout_param_list):
            pos = layout_param.get('pos')
            if interspace > 0:
                if layout_mode in ['speaker']:
                    if i > 0:
                        pos = [pos[0] + interspace, pos[1] + interspace, pos[2] - 2 * interspace, pos[3] - 2 * interspace]
                else:
                    pos = [pos[0] + interspace, pos[1] + interspace, pos[2] - 2 * interspace, pos[3] - 2 * interspace]
                layout_param['pos'] = pos

            if layout_mode in ['speaker'] and i == 0:
                if main_stream_crop_mode and main_stream_crop_mode != crop_mode:
                    layout_param['resolution_fill'] = main_stream_crop_mode

            if rotate_list:
                if len(rotate_list) > i:
                    layout_param['rotate'] = rotate_list[i]

        return layout_extra


# g_monitor_change = {
#     'frame_num': None,
#     'layout_mode': None,
#     'layout_location': None,
#     'main_stream_idx': None,
#
#     'layout_extra': None,
# }


def bg_overlay_frames(frame_list, layout_option, g_monitor_change=None):
    """
    layout_option = {
        'layout_mode': '',  # speaker  gallery
        'crop_mode': '',  # pad crop scale
        'layout_location': '',
        'interspace': 0,
        'main_stream_idx': 0,
        'main_stream_crop_mode': '',
        "width": 0,
        "height": 0,
        "background_color": "#000000",
    }
    """
    if not layout_option:
        return
    # print("[bg_overlay_frames]len(frame_list)=%s" % len(frame_list))

    width = layout_option.get('width') or 1280
    height = layout_option.get('height') or 720
    background_color = layout_option.get('background_color', "#000000")
    layout_mode = layout_option.get('layout_mode', 'speaker')
    layout_location = layout_option.get('layout_location', 'right')
    main_stream_idx = layout_option.get('main_stream_idx', 0)
    if not frame_list:
        bg_frame = create_frame(width=width, height=height, dim=3, color=background_color, store_frame=0)
        return bg_frame

    frame_num = len(frame_list)
    idx_list = list(range(0, frame_num))  # 类似帧的索引地址
    if -frame_num <= main_stream_idx < frame_num:
        main_idx = idx_list.pop(main_stream_idx)
        _new_list = [main_idx]
        _new_list.extend(idx_list)
        idx_list = _new_list

    # 如果参数没有改变 就可以重用之前的layout_extra
    if g_monitor_change:
        if frame_num != g_monitor_change.get(frame_num) or layout_mode != g_monitor_change.get('layout_mode') \
                or layout_location != g_monitor_change.get('layout_location') or main_stream_idx != g_monitor_change.get('main_stream_idx'):
                layout_extra = get_layout_extra(frame_num=frame_num, layout_option=layout_option)  # global
                g_monitor_change['layout_extra'] = copy.deepcopy(layout_extra)
                g_monitor_change['frame_num'] = frame_num
                g_monitor_change['layout_mode'] = layout_mode
                g_monitor_change['layout_location'] = layout_location
                g_monitor_change['main_stream_idx'] = main_stream_idx
        else:
            if g_monitor_change.get('layout_extra'):
                layout_extra = g_monitor_change.get('layout_extra')
            else:
                layout_extra = get_layout_extra(frame_num=frame_num, layout_option=layout_option)
    else:
        layout_extra = get_layout_extra(frame_num=frame_num, layout_option=layout_option)

    if layout_extra:
        if g_monitor_change and g_monitor_change.get('copy_frame', 0) and g_monitor_change.get('background_color') == background_color:
            bg_frame = g_monitor_change.get('background_frame').copy()
        else:
            print("create_frame(width=%s, height=%s, dim=3, color=%s) | layout_extra=%s" % (width, height, background_color, str(layout_extra)))
            bg_frame = create_frame(width=width, height=height, dim=3, color=background_color, store_frame=0)
            if g_monitor_change:
                g_monitor_change['background_frame'] = bg_frame.copy()
                g_monitor_change['background_color'] = background_color
                g_monitor_change['copy_frame'] = 1

        layout_param_list = layout_extra.get('layout_param_list')
        overlay_frame = bg_frame
        for i, layout_param in enumerate(layout_param_list):
            overlay_frame = inner_frame_add_logo(overlay_frame, frame_list[idx_list[i]], layout_param)
        return overlay_frame


###############################

def test_bg_overlay_frames():
    _starttime = time.time()
    layout_option = {
        'layout_mode': 'speaker',  # speaker  gallery
        'crop_mode': 'pad',  # pad crop scale
        'layout_location': 'right',
        'interspace': 0,
        'main_stream_idx': 0,
        'main_stream_crop_mode': 'crop',  # pad  crop  scale
        "width": 1280,
        "height": 720,
        "background_color": "#112233",
        "rotate_list": [0, 90, -90],
    }

    frame0 = create_frame(width=400, height=400, dim=3, color='#ff0000')
    frame1 = create_frame(width=400, height=300, dim=3, color='#00ff00')
    frame2 = create_frame(width=300, height=400, dim=3, color='#0000ff')
    frame3 = create_frame(width=300, height=300, dim=3, color='#ffff00')
    frame4 = create_frame(width=400, height=200, dim=3, color='#ff00ff')

    frame_list = [frame0, frame1, frame2, frame3, frame4]
    # frame_list = [frame0]
    output_frame = bg_overlay_frames(frame_list, layout_option)

    output_filename = 'test_cv2_logo_%s_%s.jpg' % (layout_option.get('main_stream_crop_mode', ''), uuid4().hex)
    cv2.imwrite(output_filename, output_frame)

    print("cost time=%s, output_filename=%s" % (str(time.time()-_starttime), output_filename))
    return output_frame


if __name__ == '__main__':
    ret = test_bg_overlay_frames()
