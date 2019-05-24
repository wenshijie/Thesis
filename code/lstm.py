# -*- coding: utf-8 -*-
"""
Created on =2019-05-23

@author: wenshijie
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import backend as k
from base_function import spilt_data


def lstm(x, y, test_num=100, spilt=0.2):
    """
    输入处理好格式的input（sample,lag,dim）,output(sample,output_dim)
    :param x: 输入
    :param y: 输出
    :param test_num:预测值的数量 优先
    :param spilt: 预测值占所有值的百分比 优先级低于test_num
    :return:
    """
    x_train, y_train, x_test, y_test, _ = spilt_data(x, y, spilt=spilt, test_num=test_num)
    k.clear_session()
    model = Sequential()
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.fit(x_train, y_train, batch_size=8, epochs=20)
    #  返回预测值（len(x_text),）numpy.array
    return np.reshape(model.predict(x_test),(len(x_test),))