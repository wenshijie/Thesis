# -*- coding: utf-8 -*-
"""
Created on =2019-05-23

@author: wenshijie
"""
import pandas as pd
import numpy as np
import time
from base_function import seq_tf_percentage, seq_tf_matrix, restore_percentage, loss_function
from pyhht.emd import EMD
from lstm import lstm

test_num = 100  # 测试数据的长度
data_use_num = 1000  # 使用的数据总长度包括训练和验证和测试
name_data = '000001'
path = '../data/'+name_data+'.csv'  # 数据的地址

df_01 = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df_01[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array

percentage = seq_tf_percentage(data)  # 转化为百分比数据

per_ = percentage[-data_use_num:]  # 取最后data_use_num个数据实验

# 分解序列
decomposer = EMD(per_)
imfs = decomposer.decompose()


def single_emd_lstm(lag=3, num_trial=20):
    x = seq_tf_matrix(imfs.T, n=lag+1)  # 转换序列，n-1个滞后项
    x = x[:, :-1, :]
    y = per_[-len(x):]

    percentage_result = []  # 百分比结果
    real_result = []  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        start_time = time.time()
        pr = lstm(x, y, test_num=test_num)  # 预测的值
        end_time = time.time()
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

        percentage_result.append(pr)
        real_result.append(restore_value)
        time_.append((end_time - start_time) / 60)  # 分钟

        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    result = {'lag': lag, 'num_sub_sequences': len(imfs), 'percentage_result': percentage_result,
              'real_result': real_result}
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/'+name_data+'single_emd_lstm_result.txt', 'a')
    fw.write(str(result) + '\n')
    fw.close()

    fw = open('../result/'+name_data+'single_emd_lstm_result_evaluation.txt', 'a')
    fw.write(str(result_evaluation) + '\n')
    fw.close()


def multi_emd_lstm(lag=3, num_trial=20):
    percentage_result = []  # 百分比结果
    real_result = []  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for j in range(num_trial):
        pr_ = None
        t = 0  # 时间累积
        for i in range(len(imfs)):
            d = seq_tf_matrix(imfs[i], n=lag+1)
            x = d[:, :-1]
            y = d[:, -1]
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # 转化成输入格式
            start_time = time.time()
            if pr_ is None:
                pr_ = lstm(x, y, test_num=test_num)  # 预测的值,子序列预测结果
            else:
                pr_ = pr_ + lstm(x, y, test_num=test_num)  # 预测的值,子序列结果直接相加
            end_time = time.time()
            t += (end_time - start_time)
        restore_value = restore_percentage(pr_, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

        percentage_result.append(pr_)
        real_result.append(restore_value)
        time_.append(t / 60)  # 分钟

        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    result = {'lag': lag, 'num_sub_sequences': len(imfs), 'percentage_result': percentage_result,
              'real_result': real_result}
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'multi_emd_lstm_result.txt', 'a')
    fw.write(str(result) + '\n')
    fw.close()

    fw = open('../result/' + name_data + 'multi_emd_lstm_result_evaluation.txt', 'a')
    fw.write(str(result_evaluation) + '\n')
    fw.close()

# single_emd_lstm(3,10)
multi_emd_lstm(3,10)