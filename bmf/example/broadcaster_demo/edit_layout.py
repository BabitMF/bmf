#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020-11-4

@author: hejianqiang
"""
import os
import copy
import json
MIN_WH = 8
"""

# LayoutConfig
layout_option = {
    'layout_mode': '',  # speaker  gallery  floating  custom
    'crop_mode': '',  # pad crop scale
    'layout_location': '',
    'interspace': 0,
    'main_stream_idx': 0,  # speaker下主流
    # 背景画布
    "width": 0,  # 没有 从上层继承
    "height": 0,
    "background_color": "#000000",
}


layout_extra = {
    'layout_mode': '',  # gallery  speaker  floating  custom
    'layout_location': '',  # right top left bottom ||for speaker mode
    'resolution_fill': '',  # pad（AspectFit） crop（AspectFill） scale（ScaleToFill）
    'interspace': 0,  # 小画面之间间隔
    'main_stream_idx': 0,  # speaker下主流

    "canvas": {
        "width": 1280,
        "height": 720,
        "color": "#000000",
      },

    'layout_param_list': [
        {
            'pos': [0, 0, 0, 0],  # [x, y, w, h]
            'location': 'top_left',
            'ref_width': 0,
            'ref_height': 0,
            'color': '#000000',
        },
        {
            'pos': [100, 0, 0, 0],  # [x, y, w, h]
            'location': 'top_left',
            'ref_width': 0,
            'ref_height': 0,
            'color': '#000000',
        },
    ]
}
"""


def create_layout_extra_gallery(frame_num, width, height, color='#000000', resolution_fill='pad', layout_location=''):
    """
    frame_info_list = [{'width', 'height', 'frame'}]
    """
    if width < MIN_WH or height < MIN_WH:
        width, height = 1280, 720

    layout_extra = {
        'layout_mode': 'gallery',  # 画廊 九宫格
        'resolution_fill': resolution_fill,
        'canvas': {'width': width, 'height': height, 'color': color}
    }

    layout_param_list = []
    layout_param_default = {'location': 'top_left', 'ref_width': width, 'ref_height': height, 'resolution_fill': resolution_fill, 'color': color}

    hw = int(width/2)  # half width
    hh = int(height/2)  # half height

    tw = int(width/3)  # one third of width
    th = int(height/3)

    qw = int(width/4)  # quarter
    qh = int(height/4)

    if frame_num == 1:
        layout_param = copy.deepcopy(layout_param_default)
        layout_param['pos'] = [0, 0, width, height]
        layout_param_list.append(layout_param)

    elif frame_num == 2:
        layout_param_0 = copy.deepcopy(layout_param_default)
        layout_param_1 = copy.deepcopy(layout_param_default)
        if width < height:
            layout_param_0['pos'] = [0, 0, width, hh]
            layout_param_1['pos'] = [0, hh, width, hh]
        else:
            layout_param_0['pos'] = [0, 0, hw, height]
            layout_param_1['pos'] = [hw, 0, hw, height]
        layout_param_list.append(layout_param_0)
        layout_param_list.append(layout_param_1)

    elif 3 <= frame_num <= 4:
        for idx in range(0, frame_num):
            j = int(idx/2)
            i = (idx % 2)
            layout_param = copy.deepcopy(layout_param_default)
            layout_param['pos'] = [i*hw, j*hh, hw, hh]
            layout_param_list.append(layout_param)

    elif 5 <= frame_num <= 9:
        for idx in range(0, frame_num):
            j = int(idx/3)
            i = (idx % 3)
            layout_param = copy.deepcopy(layout_param_default)
            layout_param['pos'] = [i*tw, j*th, tw, th]
            layout_param_list.append(layout_param)

    elif frame_num >= 10:
        for idx in range(0, min(frame_num, 16)):
            j = int(idx/4)
            i = (idx % 4)
            layout_param = copy.deepcopy(layout_param_default)
            layout_param['pos'] = [i*qw, j*qh, qw, qh]
            layout_param_list.append(layout_param)

    layout_extra['layout_param_list'] = layout_param_list
    return layout_extra


def _create_layout_extra_speaker_hor(frame_num, width, height, color='#000000', resolution_fill='pad', layout_location='right'):
    """
    主讲模式 只针对横屏
    """
    if width < MIN_WH or height < MIN_WH:
        width, height = 1280, 720

    layout_extra = {
        'layout_mode': 'speaker',  # 画廊 九宫格
        'resolution_fill': resolution_fill,
        'canvas': {'width': width, 'height': height, 'color': color}
    }

    layout_param_list = []
    layout_param_default = {'location': 'top_left', 'ref_width': width, 'ref_height': height, 'resolution_fill': resolution_fill, 'color': color}

    frame_num = min(frame_num, 13)

    layout_param_0 = copy.deepcopy(layout_param_default)
    layout_param_list.append(layout_param_0)  # 占位

    if frame_num == 1:
        layout_param_0['pos'] = [0, 0, width, height]

    else:
        if 2 <= frame_num <= 7:
            if 2 <= frame_num <= 5:
                cell_num_per_col = 4
                cell_num_per_row = 1
            else:  # if 6 <= frame_num <= 7:
                cell_num_per_col = 6
                cell_num_per_row = 1

            cell_width = int(width / (cell_num_per_col + cell_num_per_row))
            cell_height = int(height / (cell_num_per_col))

            main_width = width - cell_num_per_row * cell_width
            if layout_location in ['left']:
                layout_param_0['pos'] = [cell_num_per_row*cell_width, 0, main_width, height]
            else:
                layout_param_0['pos'] = [0, 0, main_width, height]

            for idx in range(0, frame_num-1):
                if layout_location in ['left']:
                    x = 0
                else:
                    x = width-cell_width
                y = idx*cell_height
                layout_param = copy.deepcopy(layout_param_default)
                layout_param['pos'] = [x, y, cell_width, cell_height]
                layout_param_list.append(layout_param)

        elif 8 <= frame_num <= 13:
            if 8 <= frame_num <= 9:
                cell_num_per_col = 4
                cell_num_per_row = 2
            else:
                cell_num_per_col = 6
                cell_num_per_row = 2

            cell_width = int(width / (cell_num_per_col + cell_num_per_row))
            cell_height = int(height / (cell_num_per_col))

            main_width = width-cell_num_per_row*cell_width
            if layout_location in ['left']:
                layout_param_0['pos'] = [cell_num_per_row*cell_width, 0, main_width, height]
            else:
                layout_param_0['pos'] = [0, 0, main_width, height]

            for idx in range(0, frame_num - 1):
                j = int(idx/cell_num_per_row)
                i = (idx % cell_num_per_row)
                if layout_location in ['left']:
                    x = i*cell_width
                else:
                    x = width - (cell_num_per_row-i)*cell_width
                y = j * cell_height
                layout_param = copy.deepcopy(layout_param_default)
                layout_param['pos'] = [x, y, cell_width, cell_height]
                layout_param_list.append(layout_param)

    layout_extra['layout_param_list'] = layout_param_list
    return layout_extra


def _create_layout_extra_speaker_ver(frame_num, width, height, color='#000000', resolution_fill='pad', layout_location='top'):
    """
    主讲模式 只针对横屏
    """
    if width < MIN_WH or height < MIN_WH:
        width, height = 1280, 720
    layout_extra = {
        'layout_mode': 'speaker',  # 画廊 九宫格
        'resolution_fill': resolution_fill,
        'canvas': {'width': width, 'height': height, 'color': color}
    }

    layout_param_list = []
    layout_param_default = {'location': 'top_left', 'ref_width': width, 'ref_height': height, 'resolution_fill': resolution_fill, 'color': color}

    frame_num = min(frame_num, 13)

    layout_param_0 = copy.deepcopy(layout_param_default)
    layout_param_list.append(layout_param_0)  # 占位

    if frame_num == 1:
        layout_param_0['pos'] = [0, 0, width, height]

    elif 2 <= frame_num <= 7:
        if 2 <= frame_num <= 5:
            cell_num_per_col = 1
            cell_num_per_row = 4
        else:  # if 6 <= frame_num <= 7:
            cell_num_per_col = 1
            cell_num_per_row = 6

        cell_height = int(height / (cell_num_per_col + cell_num_per_row))
        cell_width = int(width / cell_num_per_row)

        main_height = height - cell_num_per_col * cell_height
        if layout_location in ['bottom']:
            layout_param_0['pos'] = [0, 0, width, main_height]
        else:
            layout_param_0['pos'] = [0, cell_num_per_col * cell_height, width, main_height]

        for idx in range(0, frame_num-1):
            if layout_location in ['bottom']:
                y = height - cell_height
            else:
                y = 0
            x = idx*cell_width
            layout_param = copy.deepcopy(layout_param_default)
            layout_param['pos'] = [x, y, cell_width, cell_height]
            layout_param_list.append(layout_param)

    elif 8 <= frame_num <= 13:
        if 8 <= frame_num <= 9:
            cell_num_per_col = 2
            cell_num_per_row = 4
        else:
            cell_num_per_col = 2
            cell_num_per_row = 6

        cell_height = int(height / (cell_num_per_col + cell_num_per_row))
        cell_width = int(width / cell_num_per_row)

        main_height = height - cell_num_per_col * cell_height
        if layout_location in ['bottom']:
            layout_param_0['pos'] = [0, 0, width, main_height]
        else:
            layout_param_0['pos'] = [0, cell_num_per_col * cell_height, width, main_height]

        for idx in range(0, frame_num - 1):
            j = int(idx/cell_num_per_row)
            i = (idx % cell_num_per_row)
            if layout_location in ['bottom']:
                y = height - (cell_num_per_col-j)*cell_height
            else:
                y = j*cell_height
            x = idx * cell_width
            layout_param = copy.deepcopy(layout_param_default)
            layout_param['pos'] = [x, y, cell_width, cell_height]
            layout_param_list.append(layout_param)

    layout_extra['layout_param_list'] = layout_param_list
    return layout_extra


def create_layout_extra_speaker(frame_num, width, height, color='#000000', resolution_fill='pad', layout_location='right'):
    if layout_location in ['top', 'bottom']:
        layout_extra = _create_layout_extra_speaker_ver(frame_num=frame_num, width=width, height=height, color=color,
                                                        resolution_fill=resolution_fill, layout_location=layout_location)
    else:
        layout_extra = _create_layout_extra_speaker_hor(frame_num=frame_num, width=width, height=height, color=color,
                                                        resolution_fill=resolution_fill, layout_location=layout_location)
    return layout_extra


def create_layout_extra(frame_num, width, height, color='#000000', resolution_fill='pad', layout_location='right', layout_mode='speaker'):
    if layout_mode in ['speaker']:
        layout_extra = create_layout_extra_speaker(frame_num=frame_num, width=width, height=height, color=color,
                                                   resolution_fill=resolution_fill, layout_location=layout_location)
    else:
        layout_extra = create_layout_extra_gallery(frame_num=frame_num, width=width, height=height, color=color,
                                                   resolution_fill=resolution_fill, layout_location=layout_location)
    return layout_extra

