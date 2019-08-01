# -*- coding: utf-8 -*-
"""
Created on =2019-07-21

@author: wenshijie
"""
import math
from pyhht.emd import EMD
import pandas as pd
import numpy as np
from base_function import data_trans
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


name_data = '000001'
path = '../data/'+name_data+'.csv'  # 数据的地址

df_01 = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df_01[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列
df_data['Close'] = df_data['Close'].astype('float64')
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array
diff = data_trans(data)  # 转化为百分比数据
plt.figure(1)
plt.plot(diff, label='差分后序列')
plt.plot(data, linestyle='--', label='原始序列')
plt.legend()
plt.figure(figsize=(6, 9))
decomposer = EMD(diff)
imfs = decomposer.decompose()
num_imfs = imfs.shape[0]
plt.subplot(num_imfs+1, 1, 1)
plt.plot(diff)
plt.ylabel("original")
for n in range(num_imfs-1):
    plt.subplot(num_imfs+1, 1, n+2)
    plt.plot(imfs[n])
    plt.ylabel("imf %i" % (n+1))
plt.subplot(num_imfs+1, 1, num_imfs+1)
plt.plot(imfs[-1])
plt.ylabel("res")
plt.show()