# -*- coding: utf-8 -*-
"""
Created on =2019-05-23

@author: wenshijie
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras import backend as k
from base_function import spilt_data
from statsmodels.tsa.arima_model import ARMA
import sys
from matplotlib import pyplot as plt


def lstm(x, y, test_num=100, spilt=0.2, batch_size=32, epochs=20, hidden=128):
    """
    输入处理好格式的input（sample,lag,dim）,output(sample,output_dim)
    :param hidden: 隐藏层的节点数
    :param epochs: 训练多少轮，样本多时小一点，样本少时多一点
    :param batch_size: 没批次多少样本
    :param x: 输入
    :param y: 输出
    :param test_num:预测值的数量 优先
    :param spilt: 预测值占所有值的百分比 优先级低于test_num
    :return: np.array(len,)
    """
    x_train, y_train, x_test, y_test, _ = spilt_data(x, y, spilt=spilt, test_num=test_num)
    k.clear_session()
    model = Sequential()
    model.add(LSTM(hidden, dropout=0.2, recurrent_dropout=0.2, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    # 返回预测值（len(x_text),）numpy.array
    return np.reshape(model.predict(x_test), (len(x_test),))


def ann(x, y, test_num=100, spilt=0.2, hidden=128, batch_size=32, epochs=20):
    """
    输入处理好格式的input（sample,lag,dim）,output(sample,output_dim)
    :param hidden: 隐藏层的节点数
    :param epochs: 训练多少轮，样本多时小一点，样本少时多一点
    :param batch_size: 没批次多少样本
    :param x: 输入
    :param y: 输出
    :param test_num:预测值的数量 优先
    :param spilt: 预测值占所有值的百分比 优先级低于test_num
    :return: np.array(len,)
    """
    x_train, y_train, x_test, y_test, _ = spilt_data(x, y, spilt=spilt, test_num=test_num)
    k.clear_session()
    model = Sequential()
    model.add(Dense(hidden, activation='relu', input_shape=(x_train.shape[1],)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='rmsprop', loss='mse')
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    return np.reshape(model.predict(x_test), (len(x_test),))


def select_arma_model(data_ts, maxlag=15):
    """
    输入时间序列，并选择最优ARMA模型,依据BIC信息准则进行模型调优,p,q<=15
    :param data_ts: 时间序列
    :param maxlag: pq的最大值
    :return: bic_value,p,q,model
    """
    init_bic = sys.maxsize
    init_p = 0
    init_q = 0
    init_propermodel = None
    for p in np.arange(maxlag):
        for q in np.arange(maxlag):
            model = ARMA(data_ts, order=(p, q))
            try:
                results_arma = model.fit(disp=-1, method='css')
            except:
                continue
            bic = results_arma.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_propermodel = results_arma
                init_bic = bic
    return init_bic, init_p, init_q, init_propermodel


def arma_pre(const=0, ar_coefficient=[], ma_coefficient=[], series=None):
    """
    输入ARMA的系数，以及序列返回预测结果
    :param const: 常量
    :param ar_coefficient: AR的系数
    :param ma_coefficient: MA的系数
    :param series: 真实序列
    :return:
    """
    p, q = len(ar_coefficient), len(ma_coefficient)
    ar_coefficient = np.reshape(ar_coefficient, (p,))
    ma_coefficient = np.reshape(ma_coefficient, (q,))
    d = max(p, q + 1)
    pre = []
    for i in range(d, len(series)):
        y_pre = const + sum(series[i - p:i] * ar_coefficient) + sum(
            (series[i - q:i] - series[i - q - 1:i - 1]) * ma_coefficient)
        pre.append(y_pre)
    return np.reshape(pre, (len(pre),))
