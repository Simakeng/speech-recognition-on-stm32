import importlib
import os
import NN.model
import NN.utility

from keras.models import load_model

def train(data_set = None,data_per_batch=32,epoch=1,model_path='model.h5'):

    Dataset = None

    if data_set == None:
        dataloaders = os.listdir('Dataset')
        for dataloader in dataloaders:
            loader_path = os.path.join('Dataset',dataloader)
            if dataloader.endswith('.py') and os.path.isfile(loader_path) and dataloader!='__init__.py':
                try:
                    Dataset = importlib.import_module("Dataset." + dataloader[:-3])
                except:
                    print('failed to load Dataset from "%s".' % dataloader)
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

    data_loader = Dataset.DataLoader(1024,data_per_batch,13)
    
    # 加载网络模型
    model = NN.model.create_model()
    # 输出网络结构
    model.summary()
    # 加载之前训练的数据
    if(os.path.exists(model_path)):
        model = load_model(model_path)

    csv_logger = NN.utility.LossHistory('training_log.csv')

    # 开始训练
    res = model.fit_generator(
        data_loader.get_train_generator(),
        steps_per_epoch=32,
        epochs=epoch,
        validation_data=data_loader.get_validation_generator(),
        validation_steps=32,
        callbacks=[csv_logger])

    model.save(model_path)
