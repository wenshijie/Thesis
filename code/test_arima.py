# -*- coding: utf-8 -*-
"""
Created on =2019-08-26

@author: wenshijie
"""
import pandas as pd
import numpy as np
from base_function import data_trans
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


name_data = '000001'  # 数据
path = '../data/'+name_data+'.csv'  # 数据的地址
df = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列，日期向下递增，
df_data['Close'] = df_data['Close'].astype('float64')
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array
data_tf = data_trans(data)  # 数据变换
print(adfuller(data_tf))
print(acorr_ljungbox(data_tf, 4))

# (-13.317419484079407, 6.5606195062645e-25, 34, 6880, {'1%': -3.4313008345208873, '5%': -2.861960191315825, '10%': -2.566993663994896}, 70578.62953782696)
# (array([ 8.9800194 , 18.56779201, 32.70658048, 68.73587092]), array([2.72947750e-03, 9.29084455e-05, 3.71390617e-07, 4.19570537e-14]))
# (-20.52410113765169, 0.0, 17, 7104, {'1%': -3.4312708424082357, '5%': -2.861946939301189, '10%': -2.5669866097143914}, 57466.82649624165)
# (array([16.55581391, 26.15022647, 26.50217407, 28.54015245]), array([4.72389341e-05, 2.09676844e-06, 7.48606237e-06, 9.69227761e-06]))