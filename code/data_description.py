# -*- coding: utf-8 -*-
"""
Created on =2019-07-21

@author: wenshijie
"""

import pandas as pd
import numpy as np


name_data = 'sp500'
path = '../data/'+name_data+'.csv'  # 数据的地址
df = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列，日期向下递增，
df_data['Close'] = df_data['Close'].astype('float64')
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array

print('data name :{}'.format(name_data))
print('data min :{}'.format(data.min()))
print('data max :{}'.format(data.max()))
print('data mean :{}'.format(data.mean()))
print('data std :{}'.format(data.std()))

" name_data = '000001' name_data = 'sp500' 分别计算得到结果"