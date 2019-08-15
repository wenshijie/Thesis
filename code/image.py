# -*- coding: utf-8 -*-
"""
Created on =2019-06-25

@author: wenshijie
"""
import matplotlib.pyplot as plt
import json


def get_json(result_folder='result', data_name='000001', model_name='only_ann'):
    """

    :param result_folder:
    :param data_name: '000001', 'sp500'
    :param model_name: 'only_ann', 'multi_emd_ann', 'single_emd_ann'
    :return:
    """

    with open('../'+result_folder+'/'+data_name+'/'+model_name+'_result_evaluation.json', 'r') as load_f:
        readlines = load_f.readlines()
        time_, mape_, mae_, rmse_ = [], [], [], []
        time_all, mape_all, mae_all, rmse_all = [], [], [], []
        for readline in readlines:
            load_dict = json.loads(readline)
            time_.append({'lag='+str(load_dict['lag']): load_dict['time']})
            mape_.append({'lag='+str(load_dict['lag']): load_dict['mape']})
            mae_.append({'lag='+str(load_dict['lag']): load_dict['mae']})
            rmse_.append({'lag='+str(load_dict['lag']): load_dict['rmse']})

            time_all = time_all+load_dict['time']
            mape_all = mape_all+load_dict['mape']
            mae_all = mae_all+load_dict['mae']
            rmse_all = rmse_all+load_dict['rmse']
        time_.append({'all lag': time_all})
        mape_.append({'all lag': mape_all})
        mae_.append({'all lag': mae_all})
        rmse_.append({'all lag': rmse_all})
    return time_, mape_, mae_, rmse_


def get_json_arima(result_folder='result', data_name='000001', model_name='only_ann'):
    """

    :param result_folder:
    :param data_name: '000001', 'sp500'
    :param model_name: 'only_ann', 'multi_emd_ann', 'single_emd_ann'
    :return:
    """

    with open('../'+result_folder+'/'+data_name+'/'+model_name+'_result_evaluation.json', 'r') as load_f:
        readlines = load_f.readlines()
        time_, mape_, mae_, rmse_ = [], [], [], []
        time_all, mape_all, mae_all, rmse_all = [], [], [], []
        load_dict = json.loads(readlines[0])
        mape_ = load_dict['mape']
        mae_ = load_dict['mae']
        rmse_ = load_dict['rmse']
    return mape_, mae_, rmse_


# time_m, mape_m, mae_m, rmse_m = get_json(result_folder='result', data_name='000001', model_name='multi_emd_ann')
# time_s, mape_s, mae_s, rmse_s = get_json(result_folder='result', data_name='000001', model_name='single_emd_ann')
#
# plt.figure(figsize=(10, 8), dpi=256)
# for i in range(1, 9):
#     plt.subplot(2, 4, i)
#     plt.title(list(rmse_m[i-1].keys())[0])
#     plt.boxplot((list(rmse_m[i-1].values())[0], list(rmse_s[i-1].values())[0]), labels=('emd_ann', 's_emd_ann'))
# plt.show()

# time_m, mape_m, mae_m, rmse_m = get_json(result_folder='result', data_name='sp500', model_name='multi_emd_ann')
# time_s, mape_s, mae_s, rmse_s = get_json(result_folder='result', data_name='sp500', model_name='single_emd_ann')
#
# plt.figure(figsize=(10, 8), dpi=256)
# for i in range(1, 9):
#     plt.subplot(2, 4, i)
#     plt.title(list(mape_m[i-1].keys())[0])
#     plt.boxplot((list(mape_m[i-1].values())[0], list(mape_s[i-1].values())[0]), labels=('emd_ann', 's_emd_ann'))
# plt.show()

# data_name = '000001'
data_name = 'sp500'
result_folder = 'adaptive_result_ignore_0'
# 是否添加ARIMA结果线
mape_ar, mae_ar, rmse_ar = get_json_arima(result_folder=result_folder, data_name=data_name, model_name='arima')
time_o_a, mape_o_a, mae_o_a, rmse_o_a = get_json(result_folder=result_folder, data_name=data_name, model_name='only_ann')
time_m0, mape_m0, mae_m0, rmse_m0 = get_json(result_folder=result_folder, data_name=data_name, model_name='multi_emd_aann')
time_s0, mape_s0, mae_s0, rmse_s0 = get_json(result_folder=result_folder, data_name=data_name, model_name='single_emd_aann')
result_folder = 'adaptive_result_ignore_2'
time_m2, mape_m2, mae_m2, rmse_m2 = get_json(result_folder=result_folder, data_name=data_name, model_name='multi_emd_aann')
time_s2, mape_s2, mae_s2, rmse_s2 = get_json(result_folder=result_folder, data_name=data_name, model_name='single_emd_aann')

plt.figure(figsize=(12, 8), dpi=256)
for i in range(1, 9):
    plt.subplot(2, 4, i)
    plt.title(list(mape_o_a[i-1].keys())[0])
    plt.boxplot((list(mae_o_a[i-1].values())[0], list(mae_m0[i-1].values())[0], list(mae_s0[i-1].values())[0],
                 list(mae_m2[i-1].values())[0], list(mae_s2[i-1].values())[0]),
                labels=('A', 'AEA', 'SAEA', '$\mathregular{AEA^a}$', '$\mathregular{SAEA^a}$'))
    # ARIMA 结果虚线
    plt.hlines(mae_ar, 1, 5, linestyles='--', label='ARIMA')
    plt.legend()
plt.show()