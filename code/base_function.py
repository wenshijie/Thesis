# -*- coding: utf-8 -*-
"""
Created on =2019-05-23

@author: wenshijie
"""
import numpy as np
import math


def seq_tf_matrix(sequence, n=4):
    """
    把序列转换成矩阵,每行n列。得到len(sequence)-n+1个样本
    :param sequence: numpy.array,(len(sequence),dim)
    :param n: int
    :return: numpy.array,(len(sequence)-n+1,n)
    """
    sample_num = len(sequence)-n+1
    samples = []
    for i in range(sample_num):
        samples.append(sequence[i:i+n])
    samples = np.array(samples)
    return samples


def seq_tf_percentage(sequence):
    """
    把一个序列转换为变化率百分比[y(t+1)-y(t)]/y(t),返回结果长度减1
    :param sequence: numpy.array(len(sequence),)
    :return: numpy.array(len(sequence)-1,)
    """
    return (sequence[1:]-sequence[:-1])/sequence[:-1]


def restore_percentage(pre_value, lag1_value):
    """
    使用预测的百分比值和滞后一天的值还原实际预测值pre_y(t+1) = (pre_per+1)*y(t)
    :param pre_value: 预测值 numpy.array(len(pre_value),)
    :param lag1_value: 滞后值 numpy.array(len(lag1_value),)
    :return: numpy.array(len(lag1_value),)
    """
    return (pre_value+1)*lag1_value


def spilt_data(x, y, spilt=0.2, test_num=100):
    """
    分割数据分为训练集和测试集，如果没有指定多少为测试集就留下20%做测试集，如果给定测试集大小就按给定的分
    :param x: 输入数据
    :param y: 输出数据
    :param spilt:
    :param test_num:
    :return: 训练集，测试集，
    """
    if test_num:
        spilt = len(y)-test_num  # 余下test_num作为测试集
    else:
        spilt = int(len(y)*(1-spilt))
    x_train = x[:spilt]
    x_test = x[spilt:]
    y_train = y[:spilt]
    y_test = y[spilt:]
    return x_train, y_train, x_test, y_test, spilt


def mape(pre_value, real_value):
    """
    输入预测值和真实值计算MAPE
    :param pre_value: 预测值 np.array (sample,)
    :param real_value: 真实值 np.array (sample,)
    :return: float
    """
    error = pre_value-real_value
    mape_ = sum(abs(error)/real_value)/len(real_value)
    return mape_


def mae(pre_value, real_value):
    """
     输入预测值和真实值计算MAE
    :param pre_value: 预测值 np.array (sample,)
    :param real_value: 真实值 np.array (sample,)
    :return: float
    """
    error = pre_value-real_value
    mae_ = sum(abs(error))/len(real_value)
    return mae_


def mse(pre_value, real_value):
    """
    输入预测值和真实值计算MAE
    :param pre_value: 预测值 np.array (sample,)
    :param real_value: 真实值 np.array (sample,)
    :return: float
    """
    error = pre_value-real_value
    mse_ = sum(error*error)/len(real_value)
    return mse_


def rmse(pre_value, real_value):
    """
    输入预测值和真实值计算RMAE
    :param pre_value: 预测值 np.array (sample,)
    :param real_value: 真实值 np.array (sample,)
    :return: float
    """
    error = pre_value-real_value
    mse_ = sum(error*error)/len(real_value)
    return math.sqrt(mse_)


def loss_function(pre_value, real_value):
    """
    pre_value 预测值,real_value 真实值
    输入预测值和真实值计算
    """
    return mape(pre_value, real_value), mae(pre_value, real_value),\
           mse(pre_value, real_value), rmse(pre_value, real_value)


if __name__ == '__main__':
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 4])
    print(loss_function(a,b))