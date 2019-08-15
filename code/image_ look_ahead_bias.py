# -*- coding: utf-8 -*-
"""
Created on =2019-08-05

@author: wenshijie
"""
# -*- coding: utf-8 -*-
"""
Created on =2019-07-07

@author: wenshijie
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax, argrelmin
from scipy import interpolate, angle

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# Define signal
t = np.linspace(0, 1, 200)


def seq(t):
    sin = lambda x, p: np.sin(2 * np.pi * x * t + p)
    S = 3 * sin(18, 0.2) * (t - 0.2) ** 2
    S += 5 * sin(11, 2.7)
    S += 3 * sin(14, 1.6)
    S += 1 * np.sin(4 * 2 * np.pi * (t - 0.8) ** 2)
    S += t ** 2.1 - t
    return S
S = seq(t)


# 上包络线
index_max = argrelmax(S)
t_max = t[index_max]
value_max = S[index_max]
tck_upper = interpolate.splrep(t_max, value_max)
t_upper = np.linspace(t_max[0], t_max[-1], 200)
upper = interpolate.splev(t_upper, tck_upper)

# 下包络线
index_min = argrelmin(S)
t_min = t[index_min]
value_min = S[index_min]
tck_lower = interpolate.splrep(t_min, value_min)
t_lower = np.linspace(t_min[0], t_min[-1], 200)
lower = interpolate.splev(t_lower, tck_lower)

# 左镜像极值点
t_left = 2*t_min[0]-t[index_min[0][0]:index_max[0][0]+1]
t_left_value = S[index_min[0][0]:index_max[0][0]+1]

# 右镜像极值点
t_right = 2*t_max[-1]-t[index_min[0][-1]:index_max[0][-1]+1]
t_right_value = S[index_min[0][-1]:index_max[0][-1]+1]

plt.plot(t, S)
plt.plot(t_upper, upper, linestyle='-.')
plt.plot(t_lower, lower, linestyle='-.')
plt.vlines(t_max[-4], -9.0, 9.0, linestyles='dotted')
plt.annotate(r'上包络线', xy=(t_upper[50], upper[50]), xycoords='data', xytext=(+30, -30), textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.annotate(r'下包络线', xy=(t_lower[50], lower[50]), xycoords='data', xytext=(+30, -30), textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.annotate(r'T1', xy=((t_max[-4]+t_min[-3])/2, seq((t_max[-4]+t_min[-3])/2)), xycoords='data', xytext=(+20, +0), textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.scatter((t_max[-4]+t_min[-3])/2, seq((t_max[-4]+t_min[-3])/2), alpha=0.5, marker='o')

plt.annotate(r'T3', xy=(t_max[-3], value_max[-3]), xycoords='data', xytext=(+20, +0), textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))

plt.scatter(t_max[-3], value_max[-3], alpha=0.5, marker='o')

plt.annotate(r'T2', xy=(t_min[-3], value_min[-3]), xycoords='data', xytext=(+20, +0), textcoords='offset points',
             fontsize=16, arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
plt.scatter(t_min[-3], value_min[-3], alpha=0.5, marker='o')

plt.show()


