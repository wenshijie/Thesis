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
from Model import lstm, ann, proper_model, arma_pre
import json

test_num = 100  # 测试数据的长度
data_use_num = 5000  # 使用的数据总长度包括训练和验证和测试
name_data = '000001'
path = '../data/'+name_data+'.csv'  # 数据的地址

df_01 = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df_01[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array

percentage = seq_tf_percentage(data)  # 转化为百分比数据
per_ = percentage[1:]  # 取最后data_use_num个数据实验


def arima(maxlag=10):
    da = seq_tf_matrix(data, 2)
    diff = da[:, -1]-da[:, -2]
    bic, p, q, model = proper_model(diff[-data_use_num:-test_num], maxlag=maxlag)
    pr = arma_pre(model.params[0], model.arparams, model.maparams, per_[-maxlag - test_num:])[-test_num:]
    restore_value = pr+da[-test_num:,-2]  # 还原预测值
    mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
    percentage_result = pd.DataFrame(pr, columns=['per_value'])  # 百分比结果
    real_result = pd.DataFrame(restore_value, columns=['restore_value'])  # 预测重构值
    percentage_result.to_csv('../result/' + name_data + 'arima_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'arima_real_result.csv')

    result_evaluation = {'p': int(p), 'q': int(q), 'mape': mape_,
                         'mae': mae_, 'mse': mse_, 'rmse': rmse_}
    fw = open('../result/' + name_data + 'arima_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def arma(maxlag=10):
    bic, p, q, model = proper_model(per_[:-test_num], maxlag=maxlag)
    pr = arma_pre(model.params[0], model.arparams, model.maparams, per_[-maxlag-test_num:])[-test_num:]
    restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
    mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

    percentage_result = pd.DataFrame(pr, columns=['per_value'])  # 百分比结果
    real_result = pd.DataFrame(restore_value, columns=['restore_value'])  # 预测重构值
    percentage_result.to_csv('../result/' + name_data + 'arma_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'arma_real_result.csv')

    result_evaluation = {'p': int(p), 'q': int(q), 'mape': mape_,
                         'mae': mae_, 'mse': mse_, 'rmse': rmse_ }
    fw = open('../result/' + name_data + 'arma_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def only_lstm(lag=9, num_trial=2):
    x = seq_tf_matrix(per_, n=lag + 1)  # 转换序列成矩阵，n-1个滞后项，共n列
    x = x[:, :-1]
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # 转化成输入格式
    y = x[:, -1]

    percentage_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []

    for i in range(num_trial):
        start_time = time.time()
        pr = lstm(x, y, test_num=test_num)
        end_time = time.time()
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        percentage_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_only_lstm_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_only_lstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'only_lstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def only_ann(lag=3, num_trial=20, hidden=128):
    x = seq_tf_matrix(per_, n=lag + 1)  # 转换序列成矩阵，n-1个滞后项，共n列
    x = x[:, :-1]
    y = x[:, -1]

    percentage_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []

    for i in range(num_trial):
        start_time = time.time()
        pr = ann(x, y, test_num=test_num, hidden=hidden)
        end_time = time.time()
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        percentage_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_only_ann_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_only_ann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'only_ann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def single_emd_lstm(lag=3, num_trial=20):
    # 分解序列
    decomposer = EMD(per_)
    imfs = decomposer.decompose()
    x = seq_tf_matrix(imfs.T, n=lag+1)  # 转换序列成矩阵，n-1个滞后项，共n列
    x = x[:, :-1, :]
    y = per_[-len(x):]

    percentage_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        start_time = time.time()
        pr = lstm(x, y, test_num=test_num)  # 预测的值
        end_time = time.time()
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        percentage_result[str(i+1)+'_times_lag'+str(lag)] = pr
        real_result[str(i+1)+'_times_lag'+str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/'+name_data+'lag_'+str(lag)+'_single_emd_lstm_per_result.csv')
    real_result.to_csv('../result/'+name_data+'lag_'+str(lag)+'_single_emd_lstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/'+name_data+'single_emd_lstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def multi_emd_lstm(lag=3, num_trial=20):
    # 分解序列
    decomposer = EMD(per_)
    imfs = decomposer.decompose()
    percentage_result = pd.DataFrame()  # 百分比结果
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
                pr = lstm(x, y, test_num=test_num)  # 预测的值,子序列预测结果
            else:
                pr = pr + lstm(x, y, test_num=test_num)  # 预测的值,子序列结果直接相加
        end_time = time.time()
        t = (end_time - start_time)
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

        percentage_result[str(j + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(j + 1) + '_times_lag' + str(lag)] = restore_value

        time_.append(t / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_multi_emd_lstm_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_multi_emd_lstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'multi_emd_lstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def single_emd_ann(lag=3, num_trial=20,  hidden=256):
    # 分解序列
    decomposer = EMD(per_)
    imfs = decomposer.decompose()

    x = seq_tf_matrix(imfs.T, n=lag+1)  # 转换序列成矩阵，n-1个滞后项，共n列
    x = x[:, :-1, :]
    x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
    y = per_[-len(x):]

    percentage_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        start_time = time.time()
        pr = ann(x, y, test_num=test_num, hidden=hidden)  # 预测的值
        end_time = time.time()
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        percentage_result[str(i+1)+'_times_lag'+str(lag)] = pr
        real_result[str(i+1)+'_times_lag'+str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/'+name_data+'lag_'+str(lag)+'_single_emd_ann_per_result.csv')
    real_result.to_csv('../result/'+name_data+'lag_'+str(lag)+'_single_emd_ann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/'+name_data+'single_emd_ann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def multi_emd_ann(lag=3, num_trial=20, hidden=128):
    # 分解序列
    decomposer = EMD(per_)
    imfs = decomposer.decompose()
    percentage_result = pd.DataFrame()  # 百分比结果
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
            if pr is None:
                pr = ann(x, y, test_num=test_num, hidden=hidden)  # 预测的值,子序列预测结果
            else:
                pr = pr + ann(x, y, test_num=test_num, hidden=hidden)  # 预测的值,子序列结果直接相加
        end_time = time.time()
        t = (end_time - start_time)
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

        percentage_result[str(j + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(j + 1) + '_times_lag' + str(lag)] = restore_value

        time_.append(t / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_multi_emd_ann_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_multi_emd_ann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'num_sub_sequences': len(imfs), 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'multi_emd_ann_result_evaluation.json', 'a')
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


def single_emd_alstm(lag=3, num_trial=2):
    percentage_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        result = []
        start_time = time.time()
        # 100(test_num)个测试样本
        for j in range(test_num):
            if j == (test_num-1):
                x, y = get_data(per_)
            else:
                x, y = get_data(per_[:-test_num + 1 + j])
            pre = lstm(x, y, 1)
            result.append(pre[0])
        end_time = time.time()
        pr = np.array(result)
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        percentage_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的，需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_single_emd_alstm_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_single_emd_alstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'single_emd_alstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def multi_emd_alstm(lag=3, num_trial=20):
    percentage_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for j in range(num_trial):
        start_time = time.time()
        result = []
        for k in range(test_num):
            decomposer = EMD(per_[:-test_num + k])  # 最后一项不参与分解
            imfs = decomposer.decompose()  # 包括m个imf和一个res项
            pr = None
            for i in range(len(imfs)):
                d = seq_tf_matrix(np.hstack((imfs[i], [0])), n=lag + 1)  # 给imfs[i]加上一个值作为最后一项的真实值，只占个位子
                x = d[:, :-1]
                y = d[:, -1]
                x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # 转化成输入格式
                if pr is None:
                    pr = lstm(x, y, test_num=1)  # 预测的值,子序列预测结果
                else:
                    pr = pr + lstm(x, y, test_num=1)  # 预测的值,子序列结果直接相加
            result.append(pr[0])
        pr = np.array(result)
        end_time = time.time()
        t = (end_time - start_time)
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])

        percentage_result[str(j + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(j + 1) + '_times_lag' + str(lag)] = restore_value

        time_.append(t / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
    # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_multi_emd_alstm_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_multi_emd_alstm_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'multi_emd_alstm_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def single_emd_aann(lag=3, num_trial=2, hidden=128):
    percentage_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        result = []
        start_time = time.time()
        # 100(test_num)个测试样本
        for j in range(test_num):
            if j == (test_num-1):
                x, y = get_data(per_)
            else:
                x, y = get_data(per_[:-test_num + 1 + j])
            x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
            pre = ann(x, y, 1, hidden=hidden)
            result.append(pre[0])
        end_time = time.time()
        pr = np.array(result)
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        percentage_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的，需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_single_emd_aann_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_single_emd_aann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'single_emd_aann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


def multi_emd_aann(lag=3, num_trial=2, hidden=128):
    percentage_result = pd.DataFrame()  # 百分比结果
    real_result = pd.DataFrame()  # 预测重构值
    time_ = []  # 时间
    mape, mae, mse, rmse = [], [], [], []
    for i in range(num_trial):
        result = []
        start_time = time.time()
        # 100(test_num)个测试样本
        for j in range(test_num):
            decomposer = EMD(per_[:-test_num + k])  # 最后一项不参与分解
            imfs = decomposer.decompose()  # 包括m个imf和一个res项
            pr = None
            for i in range(len(imfs)):
                d = seq_tf_matrix(np.hstack((imfs[i], [0])), n=lag + 1)  # 给imfs[i]加上一个值作为最后一项的真实值，只占个位子
                x = d[:, :-1]
                y = d[:, -1]
                if pr is None:
                    pr = ann(x, y, test_num=1,hidden=hidden)  # 预测的值,子序列预测结果
                else:
                    pr = pr + ann(x, y, test_num=1,hidden=hidden)  # 预测的值,子序列结果直接相加
            result.append(pre[0])
        end_time = time.time()
        pr = np.array(result)
        restore_value = restore_percentage(pr, data[-test_num - 1:-1])  # 还原预测值
        mape_, mae_, mse_, rmse_ = loss_function(restore_value, data[-test_num:])
        # 保存第i次的结果
        percentage_result[str(i + 1) + '_times_lag' + str(lag)] = pr
        real_result[str(i + 1) + '_times_lag' + str(lag)] = restore_value
        # 保存第i次的评估指标
        time_.append((end_time - start_time) / 60)  # 分钟
        mape.append(mape_)
        mae.append(mae_)
        mse.append(mse_)
        rmse.append(rmse_)
        # 预测结果
    percentage_result['test_percentage'] = per_[-test_num:]  # 把真实的，需要预测的百分比值加入
    real_result['test_value'] = data[-test_num:]  # 把真实的需要预测的原值加入
    percentage_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_multi_emd_aann_per_result.csv')
    real_result.to_csv('../result/' + name_data + 'lag_' + str(lag) + '_multi_emd_aann_real_result.csv')
    # 预测结果评价指标
    result_evaluation = {'lag': lag, 'time': time_, 'mape': mape,
                         'mae': mae, 'mse': mse, 'rmse': rmse}

    fw = open('../result/' + name_data + 'multi_emd_aann_result_evaluation.json', 'a')
    fw.write(json.dumps(result_evaluation) + '\n')
    fw.close()


# single_emd_ann(3, 2)
# multi_emd_ann(3, 2)
single_emd_lstm(lag=3, num_trial=2)
multi_emd_lstm(lag=3, num_trial=2)
# only_lstm(lag=5, num_trial=2)
# only_ann(lag=3, num_trial=20)
# arma(maxlag=10)