#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ******************************************************
# Author        : simakeng
# Last modified : 2020/5/18 15:30
# Email         : simakeng@outlook.com
# Filename      : audio.py
# Description   : Audio process library
# ******************************************************

import numpy as np
import math


def get_frames(audio, sample_rate, frame_length=15, frame_delta=10):
    """ 
    ### GetFrame
    Desc:
    - 语音信号分帧

    Args:
    - audio: 语音信号, numpy 数组.
    - sample_rate: 信号采样率.
    - frame_length: 每帧的长度, 默认值15ms.
    - frame_delta: 每帧的帧移, 默认值10ms.

    Returns:
    - 处理好的 2 维 numpy 数组, 每行是一帧.
    """
    audio_length = len(audio)
    sample_per_frame = int(frame_length / 1000 * sample_rate)
    sample_per_delta = int(frame_delta / 1000 * sample_rate)
    frame_count = 1
    if(audio_length >= sample_per_frame):
        frame_count = int(
            math.ceil((audio_length - sample_per_frame) / sample_per_delta))

    frame_padding = frame_count * sample_per_delta + sample_per_frame - audio_length
    audio = np.concatenate((audio, np.zeros((frame_padding,))))

    index = np.tile(np.arange(0, sample_per_frame), (frame_count, 1))
    index = index + np.tile(np.arange(0, frame_count * sample_per_delta,
                                      sample_per_delta), (sample_per_frame, 1)).T
    index = index.astype(np.int)
    audio = audio[index]

    return audio


def window(samples, window_function=lambda x: x * 0 + 1):
    """ 
    ### GetFrame
    Desc:
    - 语音信号加窗

    Args:
    - samples: 语音信号, 2 维 numpy 数组, 每行是一帧.
    - window_function: 窗函数, 默认f(x) = 1

    Returns:
    - 处理好的 2 维 numpy 数组, 每行是一帧.
    """
    sample_count = samples.shape[1]
    index = window_function(np.arange(-1, 1, 2 / sample_count))
    return np.apply_along_axis(lambda arr: arr * index, 1, samples)


def pre_emphasis(audio, coefficient=0.95):
    return np.append(audio[0], audio[1:]-coefficient*audio[:-1])
