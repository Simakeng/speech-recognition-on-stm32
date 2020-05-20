from NN.feature import mfcc
from NN.model.tokens import tokenize, index_token
import wave
import os
import chardet
import numpy as np
import io

dataset_path = r'C:\Users\simak\datasets\ST-CMDS-20170001_1-OS\datas'
l = os.listdir(dataset_path)
datas_path_list = [f.split('.')[0] for f in l]
datas_path_list = list(set(datas_path_list))
max_audio_len = 0
with open('dataset.bin', 'wb') as dataset_file:
    total = len(datas_path_list)
    for i in range(total):
        item = datas_path_list[i]
        print(item, 'i: ', i, '/', total, end='\r')
        file_header = os.path.join(dataset_path, item)
        # read audio file
        audio_file = wave.open(file_header + '.wav', 'rb')
        nchannels, sampwidth, sample_rate, nframes = audio_file.getparams()[:4]
        audio_data = audio_file.readframes(nframes)
        audio_file.close()
        audio_data = np.frombuffer(audio_data, dtype=np.dtype('short'))
        seq_beg = 0
        seq_end = 0
        ids = np.argwhere(abs(audio_data) > 100)
        seq_beg = int(ids.min()) - 2000
        seq_end = int(ids.max()) + 2000
        if(seq_beg < 0):
            seq_beg = 0
        if(seq_end > audio_data.shape[0]):
            seq_end = audio_data.shape[0]
        audio_data = audio_data[seq_beg:seq_end]

        if(max_audio_len < audio_data.shape[0]):
            max_audio_len = audio_data.shape[0]
            print('max_audio_len: ', max_audio_len)

        audio_data = audio_data.astype(np.float)
        audio_data = audio_data / 32767
        audio_data = np.where(audio_data == 0, np.finfo(float).eps, audio_data)
        feats = mfcc.get_mfcc_feat(audio_data)
        feat_count = feats.shape[0]
        if(feats.shape[0] > 500):
            print(file_header, 'feats.shape[0] =', feats.shape[0],
                  ', which is greater than 500, skiped.')
            with open('special.list', 'a') as f:
                f.write(file_header + '\n')
            continue
        feats_np_array_data = b''
        with io.BytesIO() as f:
            np.save(f, feats)
            feats_np_array_data = f.getvalue()
        feat_size = len(feats_np_array_data)

        # read tokens
        lable_text = ''
        with open(file_header + '.txt', 'rb') as f:
            lable_text = f.read()
            lable_text = lable_text.decode(
                chardet.detect(lable_text)['encoding'])
        tokens = tokenize(lable_text)
        tokens = index_token(tokens)
        token_size = len(tokens)
        if(feat_count < token_size): continue
        dataset_file.write(feat_size.to_bytes(4, byteorder='big'))
        dataset_file.write(feats_np_array_data)
        dataset_file.write(token_size.to_bytes(4, byteorder='big'))
        for token in tokens:
            dataset_file.write(token.to_bytes(1, byteorder='big'))
