# -*- coding: utf-8 -*-
"""
Created on =2019-05-28

@author: wenshijie
"""
import math
from pyhht.emd import EMD
import pandas as pd
import numpy as np
from base_function import data_trans, seq_tf_matrix

def rms(seq):
    """
    计算RMS
    :param seq: np.array(sample,)
    :return:
    """
    rms_ = math.sqrt(sum(seq**2))/len(seq)
    return rms_


def theta(seq):
    decomposer = EMD(seq)
    imfs = decomposer.decompose()
    rms_imfs = sum([rms(imfs[i])**2 for i in range(len(imfs))])
    rms_original = rms(seq)
    return abs(math.sqrt(rms_imfs)-rms_original)/rms_original


def corrcoef(seq1, seq2):
    return np.corrcoef(seq1, seq2)[0, 1]


def corrcoef_imfs(seq):
    decomposer = EMD(seq)
    imfs = decomposer.decompose()
    result = [corrcoef(imfs[i], seq) for i in range(len(imfs))]
    result.append(sum(map(abs, result))/len(result))
    return result


def series_minmax(series_data):
    """
    输入序列为np.array（n,）
    输入一个序列按maxmin标准化，返回标准化的序列，以及最大值，最小值
    """
    data = series_data.copy()
    max_ = np.nanmax(data)
    min_ = np.nanmin(data)
    data = (data-min_)/(max_-min_)
    return data, max_, min_


def matrix_cor(m):
    """
    与最后一列的相关系数
    :param m:
    :return:
    """
    return list(pd.DataFrame(m).corr().iloc[-1, :])

name_data = '000001'
path = '../data/'+name_data+'.csv'  # 数据的地址

df_01 = pd.read_csv(path, encoding='gbk')  # 读取数据
df_data = df_01[['Date', 'Close']].set_index('Date').iloc[::-1]  # 把数据按日期排列
df_data['Close'] = df_data['Close'].astype('float64')
data = np.array(df_data)
data = np.reshape(data, (len(data),))  # 转换成（sample,）np.array

diff = data_trans(data)


# print(theta(series_minmax(data)[0]))
# print(theta(series_minmax(diff)[0]))

# print(corrcoef_imfs(series_minmax(data)[0]))
# print(corrcoef_imfs(series_minmax(diff)[0]))

print(matrix_cor(seq_tf_matrix(data, 10)))
# 上证指数
# [0.9925052213533833, 0.993386133667623, 0.9942571787506765, 0.9950962804128652, 0.9960115824395269,
# 0.9969224666313158, 0.9977260830415531, 0.9984645658860706, 0.9992588904110402, 1.0]
# 标普500
# [0.9980923986063942, 0.9982712262593705, 0.998458288931568, 0.9986495525239756, 0.9988430656437745,
# 0.9990563718109147, 0.9992787669886709, 0.9994977021953116, 0.9997361622154102, 1.0]
print(matrix_cor(seq_tf_matrix(diff, 10)))
# 上证指数
# [-0.008651145443581408, 0.006834999895765309, 0.021172706952804994, -0.051951892446751614, 0.003523074227788529,
#  0.07219929582013276, 0.045227720273789414, -0.037241245123885554, 0.03602161724884085, 1.0]
# 标普500
# [-0.01350694261104616, -0.017031069396570293, -0.008112646905554516, -0.003528964957479074, -0.038046959576339454,
#  -0.016926027633093563, 0.007025123013691551, -0.03670731870127638, -0.048211988418971646, 1.0]

