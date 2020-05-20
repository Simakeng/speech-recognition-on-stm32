from keras.callbacks import Callback
import os
import time


class LossHistory(Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.cepoch = 0
        self.batch_start = 0
        if(not os.path.exists(self.filename)):
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

class DataRadomizer(Callback):
    def __init__(self,generator):
        super().__init__()
        self.generator = generator
    def on_epoch_end(self, epoch, logs=None):
        self.generator.shuffle()
        return super().on_epoch_end(epoch, logs=logs)

class ModelSaver(Callback):
    def __init__(self,model,model_path,save_when_batch_end=False,save_when_epoch_end=False,save_when_training_end=True):
        super().__init__()
        self.model = model
        self.model_path = model_path
        self.save_when_batch_end = save_when_batch_end
        self.save_when_epoch_end = save_when_epoch_end
        self.save_when_training_end = save_when_training_end

    def on_batch_end(self, batch, logs=None):
        if(self.save_when_batch_end):
            self.model.save_weights(self.model_path)
        return super().on_batch_end(batch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        if((not self.save_when_batch_end)and self.save_when_epoch_end):
            self.model.save_weights(self.model_path)
        return super().on_epoch_end(epoch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        if((not self.save_when_batch_end)and(not self.save_when_epoch_end) and self.save_when_training_end):
            self.model.save_weights(self.model_path)
        return super().on_train_batch_end(batch, logs=logs)
