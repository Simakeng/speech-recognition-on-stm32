#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
import NN.feature.mfcc
import NN.model.tokens
import wave
import numpy as np
import chardet
import random
import time
import threading

random.seed(time.time())


class DataLoader():
    def __init__(self, data_tile_size, data_per_batch, feat_len):
        super().__init__()
        self.data_per_batch = data_per_batch
        self.data_tile_size = data_tile_size
        self.feat_len = feat_len
        self.dataset_path = r'dataset.bin'
        self.lock = threading.Lock()
        datasets = []
        with open(self.dataset_path, 'rb') as f:
            f.seek(0, 2)
            end_f = f.tell()
            f.seek(0, 0)
            while(f.tell() < end_f):
                array_size = int.from_bytes(f.read(4), byteorder='big')
                array_off = f.tell()
                f.seek(array_size, 1)

                token_size = int.from_bytes(f.read(4), byteorder='big')
                token_off = f.tell()
                f.seek(token_size, 1)

                datasets.append(
                    {
                        "array_size": array_size,
                        "array_off": array_off,
                        "token_size": token_size,
                        "token_off": token_off
                    })
        random.shuffle(datasets)

        num_data = len(datasets)
        self.data_train = datasets[:int(num_data*0.9)]
        self.data_vaili = datasets[int(num_data*0.9):]
        self.train_index = 0
        self.vaili_index = 0

    def get_data(self, f, train=True):

        data = {}
        with self.lock:
            if(train):
                data = self.data_train.pop()
                self.data_train.insert(0, data)
            else:
                data = self.data_vaili.pop()
                self.data_vaili.insert(0, data)

        array_size = data.get('array_size')
        array_off = data.get('array_off')
        token_size = data.get('token_size')
        token_off = data.get('token_off')

        # read lable
        f.seek(token_off, 0)
        tokens = f.read(token_size)
        tokens = np.frombuffer(tokens, dtype=np.dtype('uint8'))
        token_size = tokens.shape[0]
        if(token_size > 70):
            raise Exception()
        token_pad = 70 - token_size
        tokens = np.concatenate((tokens, np.zeros([token_pad])))

        # read data
        f.seek(array_off, 0)
        feats = f.read(array_size)
        with io.BytesIO() as buffer:
            buffer.write(feats)
            buffer.seek(0, 0)
            feats = np.load(buffer)
        array_size = feats.shape[0]
        feats_pad = 500 - feats.shape[0]
        if(feats_pad < 0):
            raise Exception()
        feats = np.concatenate((feats, np.zeros((feats_pad, 13))))

        return feats, tokens, np.array([array_size]), np.array([token_size])

    def get_batch_data(self, train=True):
        data_per_batch = self.data_per_batch
        feat_len = self.feat_len
        batch_feats = []
        batch_lables = []
        batch_feats_len = []
        batch_leble_len = []
        with open(self.dataset_path, 'rb') as f:
            for i in range(data_per_batch):
                feats, lables, feats_len, lable_len = self.get_data(
                    f=f, train=train)
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
                  'input_length': batch_feats_len,
                  'label_length': batch_leble_len}
        output = {'ctc': np.zeros([data_per_batch])}
        return inputs, output
    def shuffle(self):
        random.shuffle(self.data_vaili)
        random.shuffle(self.data_train)
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
