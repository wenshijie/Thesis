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


time_m, mape_m, mae_m, rmse_m = get_json(result_folder='result', data_name='000001', model_name='multi_emd_ann')
time_s, mape_s, mae_s, rmse_s = get_json(result_folder='result', data_name='000001', model_name='single_emd_ann')

plt.figure(figsize=(10, 8), dpi=256)
for i in range(1, 9):
    plt.subplot(2, 4, i)
    plt.title(list(rmse_m[i-1].keys())[0])
    plt.boxplot((list(rmse_m[i-1].values())[0], list(rmse_s[i-1].values())[0]), labels=('emd_ann', 's_emd_ann'))
plt.show()
"""
time_m, mape_m, mae_m, rmse_m = get_json(result_folder='result', data_name='sp500', model_name='multi_emd_ann')
time_s, mape_s, mae_s, rmse_s = get_json(result_folder='result', data_name='sp500', model_name='single_emd_ann')

plt.figure(figsize=(10, 8), dpi=256)
for i in range(1, 9):
    plt.subplot(2, 4, i)
    plt.title(list(mape_m[i-1].keys())[0])
    plt.boxplot((list(mape_m[i-1].values())[0], list(mape_s[i-1].values())[0]), labels=('emd_ann', 's_emd_ann'))
plt.show()
"""
