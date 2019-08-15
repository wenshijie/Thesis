# -*- coding: utf-8 -*-
"""
Created on =2019-06-23

@author: wenshijie
"""
import pandas as pd
import numpy as np
from Test_function import dm_test, wx_test

# data_name = '000001'
# lag = 3
# model_name = 'multi_emd_ann'
# path = '../result/'+data_name+'/real_result/lag_'+str(lag)+'_'+model_name+'_real_result.csv'
# df = pd.read_csv(path)
# df = df.iloc[:, 1:]  # 去除索引列
# data_result = np.array(df.iloc[:, :-1])
# df_mean = np.mean(data_result, axis=1)  # 求多次预测实验的均值
# df_real = np.array(df.iloc[:, -1])  # 真实值


def get_csv(result_folder='result', data_name='000001', lag=3, model_name='multi_emd_ann'):
    path = '../'+result_folder+'/' + data_name + '/real_result/lag_' + str(lag) + '_' + model_name + '_real_result.csv'
    df = pd.read_csv(path)
    df = df.iloc[:, 1:]  # 去除索引列
    data_result = np.array(df.iloc[:, :-1])
    df_mean = np.mean(data_result, axis=1)  # 求多次预测实验的均值
    df_real = np.array(df.iloc[:, -1])  # 真实值
    return df_mean, df_real


'''
#  000001 DM WX 检验
cri = 'MAPE'
df_multi = pd.DataFrame()
df_single = pd.DataFrame()
df_real = pd.DataFrame()
lags = [3, 4, 5, 6, 7, 8, 9]
print(cri)
print('lag | DM  | WX ')
for lag in lags:
    df_multi_mean, df_real = get_csv(result_folder='result', data_name='000001', lag=lag, model_name='multi_emd_ann')
    df_single_mean, _ = get_csv(result_folder='result', data_name='000001', lag=lag, model_name='single_emd_ann')
    df_multi[str(lag)] = df_multi_mean
    df_single[str(lag)] = df_single_mean
    print(' {} '.format(lag),
          "%.3f" % dm_test(df_real, df_multi_mean, df_single_mean, criteria=cri)[1],
          "%.3f" % wx_test(df_real, df_multi_mean, df_single_mean, criteria=cri)[1])
print('mean',
      "%.3f" % dm_test(df_real, df_multi.mean(axis=1), df_single.mean(axis=1), criteria=cri)[1],
      "%.3f" % wx_test(df_real, df_multi.mean(axis=1), df_single.mean(axis=1), criteria=cri)[1])
'''
data_name = '000001'
cri = 'MSE'
#  sp500 DM WX 检验
df_multi = pd.DataFrame()
df_single = pd.DataFrame()
df_real = pd.DataFrame()
lags = [3, 4, 5, 6, 7, 8, 9]
print(cri)
print('lag | DM  | WX ')
for lag in lags:
    df_multi_mean, df_real = get_csv(result_folder='result', data_name=data_name, lag=lag, model_name='multi_emd_ann')
    df_single_mean, _ = get_csv(result_folder='result', data_name=data_name, lag=lag, model_name='single_emd_ann')
    df_multi[str(lag)] = df_multi_mean
    df_single[str(lag)] = df_single_mean
    print(' {} '.format(lag),
          "%.3f" % dm_test(df_real, df_multi_mean, df_single_mean, criteria=cri)[1],
          "%.3f" % wx_test(df_real, df_multi_mean, df_single_mean, criteria=cri)[1])
print('mean',
      "%.3f" % dm_test(df_real, df_multi.mean(axis=1), df_single.mean(axis=1), criteria=cri)[1],
      "%.3f" % wx_test(df_real, df_multi.mean(axis=1), df_single.mean(axis=1), criteria=cri)[1])



