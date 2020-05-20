#! /usr/bin/env python
# -*- coding: utf-8 -*-
# ******************************************************
# Author        : simakeng
# Last modified : 2020/5/18 15:30
# Email         : simakeng@outlook.com
# Filename      : __init__.py
# Description   : Mel-frequency cepstrum feature processing library
# ******************************************************

import numpy as np
from scipy.fftpack import dct
import NN.feature.audio


def spectrum_magnitude(frames, NFFT=512):
    complex_spectrum = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spectrum)


def spectrum_power(frames, NFFT=512):
    return 1.0/NFFT * np.square(spectrum_magnitude(frames, NFFT))


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_mel_filter(num_filter=20, NFFT=512, sample_rate=16000):
    filter_points = np.linspace(
        hz2mel(0), hz2mel(sample_rate / 2), num_filter + 2)
    filter_points = mel2hz(filter_points)
    indexs = np.floor(filter_points / sample_rate * (NFFT + 1))

    flt = np.zeros((num_filter, int(NFFT / 2) + 1))
    for j in range(0, num_filter):
        for i in range(int(indexs[j]), int(indexs[j+1])):
            flt[j, i] = (i-indexs[j])/(indexs[j+1]-indexs[j])
        for i in range(int(indexs[j+1]), int(indexs[j+2])):
            flt[j, i] = (indexs[j+2]-i)/(indexs[j+2]-indexs[j+1])
    return flt


def apply_mel_filter(spectrums, num_filter=20, NFFT=512, sample_rate=16000):
    f = get_mel_filter(num_filter, NFFT, sample_rate)
    feat = np.dot(spectrums, f.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    return feat


def lifter(cepstra, L=22):
    '''Lifter function.
    Args:
        cepstra: MFCC coefficients.
        L: Numbers of lifters. Defaulted to 22.
    '''
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1+(L/2)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        return cepstra


def get_mfcc_feat(audio, feature_len=13, cep_lifter=22, sample_rate=16000, audio_window_length_ms=15, audio_window_delta_ms=10, pre_emph_coef=0.95, window_function=lambda x: x*0+1, NFFT=512, num_filters=20):
    audio = NN.feature.audio.pre_emphasis(audio, pre_emph_coef)
    frame = NN.feature.audio.get_frames(audio, sample_rate, audio_window_length_ms, audio_window_delta_ms)
    frame = NN.feature.audio.window(frame, window_function)
    specs = NN.feature.mfcc.spectrum_power(frame, NFFT)
    feats = NN.feature.mfcc.apply_mel_filter(specs, num_filters, NFFT, sample_rate)
    feats = np.log(feats)
    feats = dct(feats, type=2, axis=1, norm='ortho')[:, :feature_len]
    feats = lifter(feats, cep_lifter)
    feats = np.abs(feats)
    return feats
