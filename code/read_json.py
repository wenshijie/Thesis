# -*- coding: utf-8 -*-
"""
Created on =2019-06-13

@author: wenshijie
"""
import json


def mean(list):
    """
    求列表均值
    :param list: 列表
    :return: 均值
    """
    return sum(list)/len(list)


# with open("../result/000001/single_emd_lstm_result_evaluation.json", 'r') as load_f:
#     readlines = load_f.readlines()
#     print('lag\t'+'time\t'+'mape\t'+'mas\t'+'rmse')
#     for readline in readlines:
#         load_dict = json.loads(readline)
#         print(load_dict['lag'], mean(load_dict['time']), mean(load_dict['mape']), mean(load_dict['mae']),
#               mean(load_dict['rmse']))


def get_json(result_folder='result', data_name='000001', model_name='only_ann'):
    """

    :param result_folder:
    :param data_name: '000001', 'sp500'
    :param model_name: 'only_ann', 'multi_emd_ann', 'single_emd_ann'
    :return:
    """

    with open('../'+result_folder+'/'+data_name+'/'+model_name+'_result_evaluation.json', 'r') as load_f:
        readlines = load_f.readlines()
        print('lag '+'time '+'mape '+'mae '+'rmse '+'num_sub_sequences')
        time_, mape_, mae_, rmse_ = [], [], [], []
        for readline in readlines:
            load_dict = json.loads(readline)
            print(load_dict['lag'],
                  "%.3f" % mean(load_dict['time']),
                  "%.3f" % (mean(load_dict['mape'])*100),
                  "%.2f" % mean(load_dict['mae']),
                  "%.2f" % mean(load_dict['rmse']),
                  load_dict['num_sub_sequences'] if 'num_sub_sequences'in load_dict else 0)
            time_.append(mean(load_dict['time']))
            mape_.append(mean(load_dict['mape'])*100)
            mae_.append(mean(load_dict['mae']))
            rmse_.append(mean(load_dict['rmse']))
        print(' ',
              "%.3f" % mean(time_),
              "%.3f" % (mean(mape_)),
              "%.2f" % mean(mae_),
              "%.2f" % mean(rmse_))


'''
get_json(result_folder='result', data_name='000001', model_name='only_ann')
get_json(result_folder='result', data_name='000001', model_name='single_emd_ann')
get_json(result_folder='result', data_name='000001', model_name='single_emd_ann')
'''
get_json(result_folder='result', data_name='sp500', model_name='only_ann')
get_json(result_folder='result', data_name='sp500', model_name='multi_emd_ann')
get_json(result_folder='result', data_name='sp500', model_name='single_emd_ann')
