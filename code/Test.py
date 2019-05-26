# -*- coding: utf-8 -*-
"""
Created on =2019-05-23

@author: wenshijie
"""
# 参考https://github.com/johntwk/Diebold-Mariano-Test
from scipy.stats import t
import collections
from scipy import stats
import pandas as pd
import numpy as np


def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, criteria="MSE"):
    """

    :param actual_lst: np.array(n,)
    :param pred1_lst: np.array(n,)
    :param pred2_lst: np.array(n,)
    :param h: int default 1
    :param criteria: str "MSE" "MAPE" "MAE"
    :return: DM ,p
    """

    global d
    # Length of lists (as real numbers)
    n_t = float(len(actual_lst))

    # construct d according to crit
    if criteria == "MSE":
        d = (actual_lst-pred1_lst)**2-(actual_lst-pred2_lst)**2
    elif criteria == "MAE":
        d = abs(actual_lst - pred1_lst)-abs(actual_lst - pred2_lst)
    elif criteria == "MAPE":
        d = abs(actual_lst - pred1_lst)/actual_lst - abs(actual_lst - pred2_lst)/actual_lst
    
    # Mean of d        
    mean_d = d.mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/T)*autoCov
    gamma = []
    for lag in range(0, h):
        gamma.append(autocovariance(d, len(d), lag, mean_d))  # 0, 1, 2
    v_d = (gamma[0] + 2*sum(gamma[1:])) / n_t
    dm_stat = v_d**(-0.5)*mean_d
    harvey_adj = ((n_t + 1 - 2 * h + h * (h - 1) / n_t) / n_t) ** 0.5
    dm_stat = harvey_adj*dm_stat
    # Find p-value
    p_value = 2 * t.cdf(-abs(dm_stat), df=n_t - 1)
    
    # # Construct named tuple for return
    # dm_return = collections.namedtuple('dm_return', 'DM p_value')
    #
    # rt = dm_return(DM=dm_stat, p_value=p_value)
    
    return dm_stat, p_value


def wx_test(actual_lst, pred1_lst, pred2_lst, criteria="MSE"):

    global d1, d2
    if criteria == "MSE":
        d1 = (actual_lst-pred1_lst)**2
        d2 = (actual_lst-pred2_lst)**2
    elif criteria == "MAE":
        d1 = abs(actual_lst - pred1_lst)
        d2 = abs(actual_lst - pred2_lst)
    elif criteria == "MAPE":
        d1 = abs(actual_lst - pred1_lst)/actual_lst
        d2 = abs(actual_lst - pred2_lst)/actual_lst
    wx_p = stats.wilcoxon(d1, d2)
    return wx_p[0], wx_p[1]


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5])
    a1 = np.array([1.000001, 2.000001, 3.000001, 4.000001, 5.000001])
    a2 = np.array([1.000002, 2.000002, 3.000002, 4.000002, 5.000002])
    print(dm_test(a, a1, a2))
    print(wx_test(a, a1, a2))


