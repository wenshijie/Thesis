# -*- coding: utf-8 -*-
"""
Created on =2019-05-28

@author: wenshijie
"""
import math
from pyhht.emd import EMD
import pandas as pd
import numpy as np
from base_function import data_trans

def rms(seq):
    """
    计算RMS
    :param seq: np.array(sample,)
    :return:
    """
    rms_ = math.sqrt(sum(seq**2))/len(seq)
    return rms_


def theta(seq):
    decomposer = EMD(seq)
    imfs = decomposer.decompose()
    rms_imfs = sum([rms(imfs[i])**2 for i in range(len(imfs))])
    rms_original = rms(seq)
    return abs(math.sqrt(rms_imfs)-rms_original)/rms_original


def corrcoef(seq1, seq2):
    return np.corrcoef(seq1, seq2)[0, 1]


def corrcoef_imfs(seq):
    decomposer = EMD(seq)
    imfs = decomposer.decompose()
    result = [corrcoef(imfs[i], seq) for i in range(len(imfs))]
    result.append(sum(map(abs, result))/len(result))
    return result


def series_minmax(series_data):
    """
    输入序列为np.array（n,）
    输入一个序列按maxmin标准化，返回标准化的序列，以及最大值，最小值
    """
    data = series_data.copy()
    max_ = np.nanmax(data)
    min_ = np.nanmin(data)
    data = (data-min_)/(max_-min_)
    return data, max_, min_


name_data = 'sp500'
path = '../data/'+name_data+'.csv'  # 数据的地址

df_01 = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df_01[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列
df_data['Close'] = df_data['Close'].astype('float64')
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array

diff = data_trans(data)  # 转化为百分比数据



print(theta(series_minmax(data)[0]))
print(theta(series_minmax(diff)[0]))

# print(corrcoef_imfs(series_minmax(data)[0]))
# print(corrcoef_imfs(series_minmax(diff)[0]))
