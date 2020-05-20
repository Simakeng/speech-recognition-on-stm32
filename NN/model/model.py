from keras.engine.input_layer import Input
from keras.layers.core import Activation, Dense, Dropout, K, Lambda
from keras.layers.recurrent import GRU
from keras.engine.training import Model
from keras.layers.merge import add
from keras.optimizers import RMSprop,Adadelta
from keras.backend import ctc_batch_cost

from NN.model.tokens import get_tokens

def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def percentage_difference(y_true, y_pred):
    return K.mean(abs(y_pred/y_true - 1) * 100)

def create_pridict_model():
    tokens = get_tokens()
    num_tokens = len(tokens) + 1
    input_data = Input(name='speech_data_input', shape=(500, 13))
    layer_dense_1 = Dense(32, activation="relu", use_bias=True, kernel_initializer='he_normal')(input_data)
    layer_dropout_1 = Dropout(0.4)(layer_dense_1)
    layer_dense_2 = Dense(64, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_dropout_1)
    layer_gru1 = GRU(128, return_sequences=True, kernel_initializer='he_normal', dropout=0.4)(layer_dense_2)
    layer_gru2 = GRU(128, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', dropout=0.4)(layer_gru1)
    layer_dense_3 = Dense(128, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_gru2)
    layer_dropout_2 = Dropout(0.4)(layer_dense_3)
    layer_dense_4 = Dense(num_tokens, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_dropout_2)
    output = Activation('softmax', name='Activation0')(layer_dense_4)
    model = Model(inputs=[input_data], outputs=output)
    model.compile(loss={'Activation0': lambda y_true, output: output}, optimizer='sgd')
    print("model compiled successful!")
    return model

def create_model():
    tokens = get_tokens()
    num_tokens = len(tokens) + 1
    input_data = Input(name='speech_data_input', shape=(500, 13))
    layer_dense_1 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(input_data)
    layer_dropout_1 = Dropout(0.4)(layer_dense_1)
    layer_dense_2 = Dense(512, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_dropout_1)
    layer_gru1 = GRU(512, return_sequences=True, kernel_initializer='he_normal', dropout=0.4)(layer_dense_2)
    layer_gru2 = GRU(512, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', dropout=0.4)(layer_gru1)
    layer_dense_3 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_gru2)
    layer_dropout_2 = Dropout(0.4)(layer_dense_3)
    layer_dense_4 = Dense(num_tokens, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_dropout_2)
    output = Activation('softmax', name='Activation0')(layer_dense_4)
    #ctc
    labels = Input(name='speech_labels', shape=[70], dtype='int64')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([labels, output, input_length, label_length])
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    adad = Adadelta(lr=0.01, rho=0.95, epsilon=K.epsilon())
    model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=adad)
    print("model compiled successful!")
    return model



