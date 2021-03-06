# -*- coding: utf-8 -*-
"""
Created on =2019-05-23

@author: wenshijie
"""
import pandas as pd
import numpy as np
import time
from base_function import data_trans, seq_tf_matrix, restore_data, loss_function
from pyhht.emd import EMD
from Model import lstm, ann, select_arma_model, arma_pre
import json
import os

batch_size = 32  # 每多少个样本更新一下参数
test_num = 50  # 测试数据的长度
# data_use_num = 1000  # 使用数据的长度可以只使用一定量的数据
name_data = 'sp500'  # 数据
path = '../data/'+name_data+'.csv'  # 数据的地址

if not os.path.exists('../result/'+name_data):  # 对应数据结果保存
    os.makedirs('../result/'+name_data)
if not os.path.exists('../result/'+name_data+'/data_tf_result'):  # 数据转换预测结果保存
    os.makedirs('../result/'+name_data+'/data_tf_result')
if not os.path.exists('../result/'+name_data+'/real_result'):  # 还原预测结果保存
    os.makedirs('../result/'+name_data+'/real_result')

df = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列，日期向下递增，
df_data['Close'] = df_data['Close'].astype('float64')
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array

data_tf = data_trans(data)  # 数据变换
# data_tf = data_tf[-data_use_num:]  # 取最后data_use_num个数据实验

# 分解序列
decomposer = EMD(data_tf)
imfs = decomposer.decompose()


def arima(maxlag=10):
    bic, p, q, model = select_arma_model(data_tf[:-test_num], maxlag=maxlag)
    pr = arma_pre(model.params[0], model.arparams, model.maparams, data_tf[-maxlag-test_num:])[-test_num:]
    restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
    mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

    pre_data_tf_result = pd.DataFrame(pr, columns=['data_pre_value'])  # 变换数据的预测值
    real_result = pd.DataFrame(restore_value, columns=['restore_value'])  # 预测重构值
    pre_data_tf_result.to_csv('../result/' + name_data + '/arima_data_tf_result.csv')
    real_result.to_csv('../result/' + name_data + '/arima_real_result.csv')

    result_evaluation = {'p': int(p), 'q': int(q), 'mape': mape_,
                         'mae': mae_, 'mse': mse_, 'rmse': rmse_ }
    fw = open('../result/' + name_data + '/arima_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def only_lstm(lag=3, num_trial=20, hidden=128, epochs=20):
    x = seq_tf_matrix(data_tf, n=lag + 1)  # 转换序列成矩阵，n-1个滞后项，共n列
    x = x[:, :-1]
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # 转化成输入格式
    y = x[:, -1]

    pre_data_tf_result = pd.DataFrame()  # 变换数据预测结果
    real_result = pd.DataFrame()  # 变换数据预测结果重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []

    for i in range(num_trial):
        start_time = time.time()
        pr = lstm(x, y, test_num=test_num, batch_size=batch_size, epochs=epochs, hidden=hidden)
        end_time = time.time()
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        pre_data_tf_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../result/' + name_data + '/data_tf_result/lag_' + str(lag) + '_only_lstm_data_tf_result.csv')
    real_result.to_csv('../result/' + name_data + '/real_result/lag_' + str(lag) + '_only_lstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + '/only_lstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def only_ann(lag=3, num_trial=20, hidden=128, epochs=20):
    x = seq_tf_matrix(data_tf, n=lag + 1)  # 转换序列成矩阵，n-1个滞后项，共n列
    x = x[:, :-1]
    y = x[:, -1]

    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []

    for i in range(num_trial):
        start_time = time.time()
        pr = ann(x, y, test_num=test_num, hidden=hidden, batch_size=batch_size, epochs=epochs)
        end_time = time.time()
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        pre_data_tf_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../result/' + name_data + '/data_tf_result/lag_' + str(lag) + '_only_ann_data_tf_result.csv')
    real_result.to_csv('../result/' + name_data + '/real_result/lag_' + str(lag) + '_only_ann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + '/only_ann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def single_emd_lstm(lag=3, num_trial=20, hidden=256, epochs=20):
    x = seq_tf_matrix(imfs.T, n=lag+1)  # 转换序列成矩阵，n-1个滞后项，共n列
    x = x[:, :-1, :]
    y = data_tf[-len(x):]

    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        start_time = time.time()
        pr = lstm(x, y, test_num=test_num, batch_size=batch_size, epochs=epochs, hidden=hidden)  # 预测的值
        end_time = time.time()
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        pre_data_tf_result[str(i+1)+'_times_lag'+str(lag)] = pr
        real_result[str(i+1)+'_times_lag'+str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../result/'+name_data+'/data_tf_result/lag_'+str(lag)+'_single_emd_lstm_data_tf_result.csv')
    real_result.to_csv('../result/'+name_data+'/real_result/lag_'+str(lag)+'_single_emd_lstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/'+name_data+'/single_emd_lstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def multi_emd_lstm(lag=3, num_trial=20, hidden=128, epochs=20):

    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for j in range(num_trial):
        pr = None
        start_time = time.time()
        for i in range(len(imfs)):
            d = seq_tf_matrix(imfs[i], n=lag+1)
            x = d[:, :-1]
            y = d[:, -1]
            x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # 转化成输入格式
            if pr is None:
                pr = lstm(x, y, test_num=test_num, batch_size=batch_size, epochs=epochs, hidden=hidden)  # 预测的值,子序列预测结果
            else:
                pr = pr + lstm(x, y, test_num=test_num, batch_size=batch_size, epochs=epochs, hidden=hidden)  # 预测的值,子序列结果直接相加
        end_time = time.time()
        t = (end_time - start_time)
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

        pre_data_tf_result[str(j + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(j + 1) + '_times_lag' + str(lag)] = restore_value

        time_.append(t / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../result/' + name_data + '/data_tf_result/lag_' + str(lag) + '_multi_emd_lstm_data_tf_result.csv')
    real_result.to_csv('../result/' + name_data + '/real_result/lag_' + str(lag) + '_multi_emd_lstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + '/multi_emd_lstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def single_emd_ann(lag=3, num_trial=20,  hidden=256, epochs=20):

    x = seq_tf_matrix(imfs.T, n=lag+1)  # 转换序列成矩阵，n-1个滞后项，共n列
    x = x[:, :-1, :]
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
    y = data_tf[-len(x):]

    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        start_time = time.time()
        pr = ann(x, y, test_num=test_num, hidden=hidden, batch_size=batch_size, epochs=epochs)  # 预测的值
        end_time = time.time()
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        pre_data_tf_result[str(i+1)+'_times_lag'+str(lag)] = pr
        real_result[str(i+1)+'_times_lag'+str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../result/'+name_data+'/data_tf_result/lag_'+str(lag)+'_single_emd_ann_data_tf_result.csv')
    real_result.to_csv('../result/'+name_data+'/real_result/lag_'+str(lag)+'_single_emd_ann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/'+name_data+'/single_emd_ann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def multi_emd_ann(lag=3, num_trial=20, hidden=128, epochs=20):

    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for j in range(num_trial):
        pr = None
        start_time = time.time()
        for i in range(len(imfs)):
            d = seq_tf_matrix(imfs[i], n=lag+1)
            x = d[:, :-1]
            y = d[:, -1]
            if pr is None:  # 预测的值,子序列预测结果
                pr = ann(x, y, test_num=test_num, hidden=hidden, batch_size=batch_size, epochs=epochs)
            else:  # 预测的值,子序列结果直接相加
                pr = pr + ann(x, y, test_num=test_num, hidden=hidden, batch_size=batch_size, epochs=epochs)
        end_time = time.time()
        t = (end_time - start_time)
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

        pre_data_tf_result[str(j + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(j + 1) + '_times_lag' + str(lag)] = restore_value

        time_.append(t / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../result/' + name_data + '/data_tf_result/lag_' + str(lag) + '_multi_emd_ann_data_tf_result.csv')
    real_result.to_csv('../result/' + name_data + '/real_result/lag_' + str(lag) + '_multi_emd_ann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + '/multi_emd_ann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


# arima(15)
# lag = [3, 4, 5, 6, 7, 8, 9]
# for lag in lag:
#     only_ann(lag, 2)
#     only_lstm(lag, 2)
#     single_emd_ann(lag, 2)
#     single_emd_lstm(lag, 2)
#     multi_emd_ann(lag, 2)
#     multi_emd_lstm(lag, 2)
# single_emd_lstm(3, 2, epochs=20)
# single_emd_lstm(3, 5, epochs=30)
# single_emd_ann(3, 1)