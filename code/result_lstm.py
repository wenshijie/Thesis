# -*- coding: utf-8 -*-
"""
Created on =2019-05-23

@author: wenshijie
"""
import pandas as pd
import numpy as np
from base_function import seq_tf_percentage, seq_tf_matrix, restore_percentage, loss_function
# from pyhht.emd import EMD
from lstm import lstm
from PyEMD import EMD

path = '../data/000001.csv'
df_01 = pd.read_csv(path, encoding='gbk')
df_data = df_01[['Date', 'Close']].set_index('Date').iloc[::-1]
data = np.array(df_data)
per = seq_tf_percentage(data)
# decomposer = EMD(np.reshape(per[-1000:], (1000,)))
# imfs = decomposer.decompose()
emd = EMD()
imfs = emd(np.reshape(per[-1000:], (1000,)))
x = seq_tf_matrix(imfs.T, n=4)  # 转换序列，n-1个滞后项
x = x[:, :-1, :]
y = per[-len(x):]
result1 = []
for i in range(20):
    pr = lstm(x, y)
    result1.append(pr)


# re1 = []
# for i in range(20):
#     re,*_ = loss_function(restore_percentage(result1[i],np.reshape(data[-101:-1],(len(data[-101:-1]),))),np.reshape(data[-100:],(len(data[-100:]),)))
#     re1.append(re)
