# -*- coding: utf-8 -*-
"""
Created on =2019-05-23

@author: wenshijie
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

def lstm(x, y):
    """
    输入处理好格式的input（sample,lag,dim）,output(sample,output_dim)
    
    """

    spilt = -100
    x_train = x[:spilt]
    x_test = x[spilt:]
    y_train = y[:spilt]
    y_test = y[spilt:]
    model = Sequential()
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5,input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(x_train, y_train, batch_size=8, epochs=20, validation_data=(x_test, y_test))
    #  返回预测值（len(x_text),）numpy.array x_train.shape[2]
    return np.reshape(model.predict(x_test),(len(x_test),))