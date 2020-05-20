import importlib
import os

import NN.model
import NN.utility

from keras.models import load_model
from keras.layers.core import K
def predict(data_set = None,data_per_batch=32,epoch=1,model_path='model.h5'):

    Dataset = None

    if data_set == None:
        dataloaders = os.listdir('Dataset')
        for dataloader in dataloaders:
            loader_path = os.path.join('Dataset',dataloader)
            if dataloader.endswith('.py') and os.path.isfile(loader_path) and dataloader!='__init__.py':
                try:
                    Dataset = importlib.import_module("Dataset." + dataloader[:-3])
                except Exception as ex:
                    print('failed to load Dataset from "%s".' % dataloader,ex)
                else:
                    print('successfuly loaded Dataset from "%s"!' % dataloader)
                    break
        if Dataset == None:
            raise Exception('No vaild dataset found!')
    else:
        try:
            Dataset = importlib.import_module("Dataset." + data_set)
        except Exception as ex:
            raise Exception('"%s" is not a vaild dataset!' % data_set)

    data_loader = Dataset.DataLoader(1024,1,13)
    
    # 加载网络模型
    model = NN.model.create_pridict_model()
    # 输出网络结构
    model.summary()
    # 加载之前训练的数据
    model.load_weights(model_path)
    # 验证集
    validation_data = data_loader.get_validation_generator()
    data = next(validation_data)[0]
    r = model.predict(data['speech_data_input'])
    r = K.ctc_decode(r,data['input_length'][0])
    r1 = K.get_value(r[0][0])
    r1 = r1[0]

    tokens = NN.model.get_tokens()

    print('predict: [',end='')
    for i in r1:
        print(tokens[i],end=', ')
    print(']')
    print('truth  : [',end='')
    for i in range(data['label_length'][0][0]):
        print(tokens[int(data['speech_labels'][0][i])],end=', ')
    print(']')
    pass
