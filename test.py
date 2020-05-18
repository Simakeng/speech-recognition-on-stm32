import NN.feature.audio
import NN.feature.mfcc
import wave
import math
import numpy as np

wav = wave.open(r'D:\Repos\Speech Recognition in Cpp\Speech Recognition in Cpp\Audio\20170001P00001A0001.wav','rb')
params = wav.getparams()  
nchannels, sampwidth, sample_rate, nframes = params[:4]
audio = wav.readframes(nframes)  
wav.close()

a_0 = 0.53836
hamming_window = lambda t : a_0 - (1 - a_0) * np.cos(2 * np.pi * t)


audio = np.fromstring(audio,dtype=np.dtype('short'))
audio = audio.astype(np.float)
audio = audio / 32767

feats = NN.feature.mfcc.get_mfcc_feat(audio)

with open('test.xls','w') as f:          
    h,w = feats.shape
    for i in range(h):
        for j in range(w):
            val = feats[i,j]
            f.write("%f," % val)
        f.write('\n')


pass
