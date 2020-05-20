#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import NN.feature.mfcc
import NN.model.tokens
import wave
import numpy as np
import chardet
import random
import time

random.seed(time.time())


class DataLoader():
    def __init__(self, data_tile_size, data_per_batch, feat_len):
        super().__init__()
        self.data_per_batch = data_per_batch
        self.data_tile_size = data_tile_size
        self.feat_len = feat_len
        self.dataset_path = r'C:\Users\simak\datasets\ST-CMDS-20170001_1-OS\datas'
        datas_path_list = []
        if(os.path.exists('data.list')):
            with open('data.list') as f:
                datas_path_list = f.read().split('\n')
        else:
            l = os.listdir(self.dataset_path)
            datas_path_list = [f.split('.')[0] for f in l]
            datas_path_list = list(set(datas_path_list))
            with open('data.list','w') as f:
                f.write('\n'.join(datas_path_list))

        random.shuffle(datas_path_list)
        num_datas = len(datas_path_list)

        self.data_train = datas_path_list[:int(num_datas / 10 * 9)]
        self.data_vaili = datas_path_list[int(num_datas / 10 * 9):]


    def get_data(self,train=True):

        file_name = ''
        if(train):
            file_name = self.data_train.pop()
            self.data_train.append(file_name)
        else:
            file_name = self.data_vaili.pop()
            self.data_vaili.append(file_name)

        # read lable
        lable_text = ''
        with open(os.path.join(self.dataset_path, file_name) + '.txt', 'rb') as f:
            lable_text = f.read()
        lable_text = lable_text.decode(chardet.detect(lable_text)['encoding'])
        tokens = NN.model.tokens.tokenize(lable_text)
        lables = NN.model.tokens.index_token(tokens)
        # read data
        wav = wave.open(os.path.join(self.dataset_path, file_name) + '.wav', 'rb')
        nchannels, sampwidth, sample_rate, nframes = wav.getparams()[:4]
        audio_data = wav.readframes(nframes)
        wav.close()
        audio_data = np.frombuffer(audio_data, dtype=np.dtype('short'))
        audio_data = audio_data.astype(np.float)
        audio_data = audio_data / 32767
        audio_data = np.where(audio_data == 0, np.finfo(float).eps, audio_data)
        feats = NN.feature.mfcc.get_mfcc_feat(audio_data)
        feats_len = feats.shape[0]
        tile_size = self.data_tile_size - feats_len
        last_feat = feats[-1]
        pad_data = np.tile(last_feat, (tile_size, 1))
        feats = np.concatenate((feats, pad_data))
        lable_len = len(lables)
        pad_lable = 100 - lable_len
        for i in range(pad_lable):
            lables.append(0)
        return feats, np.array(lables), np.array([feats_len]),np.array([lable_len]) 

    def get_batch_data(self,train=True):
        data_per_batch = self.data_per_batch
        feat_len = self.feat_len
        batch_feats = []
        batch_lables = []
        batch_feats_len = []
        batch_leble_len = []
        for i in range(data_per_batch):
            feats, lables, feats_len, lable_len = self.get_data(train)
            batch_feats.append(feats)
            batch_lables.append(lables)
            batch_feats_len.append(feats_len)
            batch_leble_len.append(lable_len)

        batch_feats = np.array(batch_feats)
        batch_lables = np.array(batch_lables)
        batch_feats_len = np.array(batch_feats_len)
        batch_leble_len = np.array(batch_leble_len)

        inputs = {'speech_data_input': batch_feats,
                  'speech_labels': batch_lables,
                  'input_length' : batch_feats_len,
                  'label_length' : batch_leble_len}
        output = {'ctc' : np.zeros([data_per_batch])}
        return inputs, output

    def get_train_generator(self):
        def generator():
            while True:
                yield self.get_batch_data(train=True)
        return generator()

    def get_validation_generator(self):
        def generator():
            while True:
                yield self.get_batch_data(train=False)
        return generator()
