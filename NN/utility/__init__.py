from keras.callbacks import Callback
import os
import time


class LossHistory(Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.cepoch = 0
        self.batch_start = 0
        with open(self.filename, 'w') as f:
            f.write('time,epoch,batch,loss,batch_time,\n')

    def on_epoch_begin(self, epoch, logs={}):
        self.cepoch = epoch
        return super().on_epoch_begin(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.batch_start = time.time()
        return super().on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs={}):
        with open(self.filename, 'a') as f:
            str_time = time.strftime("%Y/%m/%d %H:%M:%S")
            cepoch = self.cepoch
            cbatch = logs['batch']
            closs = logs['loss']
            batch_time = time.time() - self.batch_start
            f.write("%s,%d,%d,%f,%f,\n" %
                    (str_time, cepoch, cbatch, closs, batch_time))
