# -*- coding: utf-8 -*-
"""
Created on =2019-06-11

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
name_data = '000001'
# data_use_num = 1000
ignore = 0  # 是否忽略最近的n项
path = '../data/'+name_data+'.csv'  # 数据的地址
ada_result = 'adaptive_result_ignore_'+str(ignore)
if not os.path.exists('../'+ada_result+'/'+name_data):
    os.makedirs('../'+ada_result+'/'+name_data)
if not os.path.exists('../'+ada_result+'/'+name_data+'/data_tf_result'):  # 数据转换预测结果保存
    os.makedirs('../'+ada_result+'/'+name_data+'/data_tf_result')
if not os.path.exists('../'+ada_result+'/'+name_data+'/real_result'):  # 还原预测结果保存
    os.makedirs('../'+ada_result+'/'+name_data+'/real_result')

df = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列，日期向下递增，
df_data['Close'] = df_data['Close'].astype('float64')
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array

data_tf = data_trans(data)  # 数据变换
# data_tf = data_tf[-data_use_num:]  # 取最后data_use_num个数据实验


def arima(maxlag=10):
    bic, p, q, model = select_arma_model(data_tf[:-test_num], maxlag=maxlag)
    pr = arma_pre(model.params[0], model.arparams, model.maparams, data_tf[-maxlag-test_num:])[-test_num:]
    restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
    mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

    pre_data_tf_result = pd.DataFrame(pr, columns=['data_pre_value'])  # 变换数据的预测值
    real_result = pd.DataFrame(restore_value, columns=['restore_value'])  # 预测重构值
    pre_data_tf_result.to_csv('../'+ada_result+'/' + name_data + '/arima_data_tf_result.csv')
    real_result.to_csv('../'+ada_result+'/' + name_data + '/arima_real_result.csv')

    result_evaluation = {'p': int(p), 'q': int(q), 'mape': mape_,
                         'mae': mae_, 'mse': mse_, 'rmse': rmse_ }
    fw = open('../'+ada_result+'/' + name_data + '/arima_result_evaluation.json', 'a')
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
    pre_data_tf_result.to_csv('../'+ada_result+'/' + name_data + '/data_tf_result/lag_' + str(lag) + '_only_lstm_data_tf_result.csv')
    real_result.to_csv('../'+ada_result+'/' + name_data + '/real_result/lag_' + str(lag) + '_only_lstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../'+ada_result+'/' + name_data + '/only_lstm_result_evaluation.json', 'a')
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
    pre_data_tf_result.to_csv('../'+ada_result+'/' + name_data + '/data_tf_result/lag_' + str(lag) + '_only_ann_data_tf_result.csv')
    real_result.to_csv('../'+ada_result+'/' + name_data + '/real_result/lag_' + str(lag) + '_only_ann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../'+ada_result+'/' + name_data + '/only_ann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def get_data(l, lag=3):
    """
    默认滞后3项，预测一步
    l 的最后一项不参与分解
    """
    decomposer = EMD(l[:-1])  # l的最后一项不参与分解
    imfs = decomposer.decompose()  # 包括m个imf和一个res项
    #  得到如下的输入样本，第一个样本（1，lag，m+1），即lag个滞后项，每一项有m+1个元素
    #  [[imf1_1,imf2_1,...,imfm_1,res_1],[imf1_2,imf2_2,...,imfm_2,res_2],...,[imf1_lag,imf2_lag,...,imfm_lag,res_lag]]
    x = seq_tf_matrix(imfs.T, lag)
    #  y为输出结果，未来一步的预测值
    y = l[-len(x):]
    return x, y


def single_emd_alstm(lag=3, num_trial=2, hidden=256, epochs=20, ignore=ignore):
    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        result = []
        start_time = time.time()
        # 100(test_num)个测试样本
        for j in range(test_num):
            if j == (test_num-1):
                x, y = get_data(data_tf)
            else:
                x, y = get_data(data_tf[:-test_num + 1 + j])
            if ignore:
                x = x[:, :-ignore, :]  # 忽略与预测值最近的ignore项
            pre = lstm(x, y, 1, batch_size=batch_size, hidden=hidden, epochs=epochs)
            result.append(pre[0])
        end_time = time.time()
        pr = np.array(result)
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        pre_data_tf_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的，需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../'+ada_result+'/' + name_data + '/data_tf_result/lag_' + str(lag) +
                              '_single_emd_alstm_data_tf_result.csv')
    real_result.to_csv('../'+ada_result+'/' + name_data + '/real_result/lag_' + str(lag) +
                       '_single_emd_alstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../'+ada_result+'/' + name_data + '/single_emd_alstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def multi_emd_alstm(lag=3, num_trial=20, hidden=128,  epochs=20, ignore=ignore):
    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for j in range(num_trial):
        start_time = time.time()
        result = []
        for k in range(test_num):
            decomposer = EMD(data_tf[:-test_num + k])  # 最后一项不参与分解
            imfs = decomposer.decompose()  # 包括m个imf和一个res项
            pr = None
            for i in range(len(imfs)):
                d = seq_tf_matrix(np.hstack((imfs[i], [0])), n=lag + 1)  # 给imfs[i]加上一个值作为最后一项的真实值，只占个位子作为要预测的值
                x = d[:, :-1]
                y = d[:, -1]
                x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # 转化成输入格式
                if ignore:
                    x = x[:, :-ignore, :]  # 忽略与预测值最近的ignore项
                if pr is None:
                    pr = lstm(x, y, test_num=1, batch_size=batch_size, hidden=hidden, epochs=epochs)  # 预测的值,子序列预测结果
                else:
                    pr = pr + lstm(x, y, test_num=1, batch_size=batch_size, hidden=hidden, epochs=epochs)  # 预测的值,子序列结果直接相加
            result.append(pr[0])
        pr = np.array(result)
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
    pre_data_tf_result.to_csv('../'+ada_result+'/' + name_data + '/data_tf_result/lag_' + str(lag) +
                              '_multi_emd_alstm_data_tf_result.csv')
    real_result.to_csv('../'+ada_result+'/' + name_data + '/real_result/lag_' + str(lag) +
                       '_multi_emd_alstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../'+ada_result+'/' + name_data + '/multi_emd_alstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def single_emd_aann(lag=3, num_trial=2, hidden=256,  epochs=20, ignore=ignore):
    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        result = []
        start_time = time.time()
        # 100(test_num)个测试样本
        for j in range(test_num):
            if j == (test_num-1):
                x, y = get_data(data_tf)
            else:
                x, y = get_data(data_tf[:-test_num + 1 + j])
            if ignore:
                x = x[:, :-ignore, :]  # 忽略与预测值最近的ignore项
            x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
            pre = ann(x, y, 1, hidden=hidden, batch_size=batch_size, epochs=epochs)
            result.append(pre[0])
        end_time = time.time()
        pr = np.array(result)
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        pre_data_tf_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的，需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../'+ada_result+'/' + name_data + '/data_tf_result/lag_' + str(lag) +
                              '_single_emd_aann_data_tf_result.csv')
    real_result.to_csv('../'+ada_result+'/' + name_data + '/real_result/lag_' + str(lag) +
                       '_single_emd_aann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../'+ada_result+'/' + name_data + '/single_emd_aann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def multi_emd_aann(lag=3, num_trial=2, hidden=128,  epochs=20, ignore=ignore):
    pre_data_tf_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for j in range(num_trial):
        result = []
        start_time = time.time()
        # 100(test_num)个测试样本
        for k in range(test_num):
            decomposer = EMD(data_tf[:-test_num + k])  # 最后一项不参与分解
            imfs = decomposer.decompose()  # 包括m个imf和一个res项
            pr = None
            for i in range(len(imfs)):
                d = seq_tf_matrix(np.hstack((imfs[i], [0])), n=lag + 1)  # 给imfs[i]加上一个值作为最后一项的真实值，只占个位子
                x = d[:, :-1]
                if ignore:
                    x = x[:, :-ignore]  # 忽略与预测值最近的ignore项
                y = d[:, -1]
                if pr is None:
                    pr = ann(x, y, test_num=1, batch_size=batch_size, hidden=hidden, epochs=epochs)  # 预测的值,子序列预测结果
                else:
                    pr = pr + ann(x, y, test_num=1, batch_size=batch_size, hidden=hidden, epochs=epochs)  # 预测的值,子序列结果直接相加
            result.append(pr[0])
        end_time = time.time()
        pr = np.array(result)
        restore_value = restore_data(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        pre_data_tf_result[str(j + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(j + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    pre_data_tf_result['test_percentage'] = data_tf[-test_num:]  # 把真实的，需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    pre_data_tf_result.to_csv('../'+ada_result+'/' + name_data + '/data_tf_result/lag_' + str(lag) +
                              '_multi_emd_aann_data_tf_result.csv')
    real_result.to_csv('../'+ada_result+'/' + name_data + '/real_result/lag_' + str(lag) +
                       '_multi_emd_aann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../'+ada_result+'/' + name_data + '/multi_emd_aann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


arima(15)
lags = [5, 6, 7, 8, 9, 10, 11]
for lag in lags:
    only_ann(lag=lag, num_trial=20)
    single_emd_aann(lag=lag, num_trial=20)
    multi_emd_aann(lag=lag, num_trial=20)
# only_ann(9, 2)
# only_lstm(9, 2)
# single_emd_aann(9, 1)
# single_emd_alstm(9, 1)
# only_lstm(9, 2)
# only_ann(9, 2)
# single_emd_alstm(9, 2, ignore=2)
# single_emd_aann(9, 2, ignore=2)
# multi_emd_alstm(9, 2, ignore=2)
# multi_emd_aann(9, 2, ignore=2)
