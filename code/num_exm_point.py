# -*- coding: utf-8 -*-
"""
Created on =2019-07-21

@author: wenshijie
"""
# 计算序列极值点个数

import pandas as pd
import numpy as np
from base_function import data_trans
from scipy.signal import argrelmax, argrelmin


name_data = '000001'
path = '../data/'+name_data+'.csv'  # 数据的地址
df = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列，日期向下递增，
df_data['Close'] = df_data['Close'].astype('float64')
data = np.array(df_data)
diff_data = data_trans(data)

# 原序列极值点
print('原序列极小值点的个数：{}'.format(len(argrelmin(data)[0])))
print('原序列极大值点的个数：{}'.format(len(argrelmax(data)[0])))
print('原序列极值点的个数：{}'.format(len(argrelmin(data)[0])+len(argrelmax(data)[0])))
# 差分后序列极值点
print('差分后序列极小值点的个数：{}'.format(len(argrelmin(diff_data)[0])))
print('差分后序列极大值点的个数：{}'.format(len(argrelmax(diff_data)[0])))
print('差分后序列极值点的个数：{}'.format(len(argrelmin(diff_data)[0])+len(argrelmax(diff_data)[0])))
