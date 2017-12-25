
'''

Feature Selection

refer to:
http://blog.csdn.net/bryan__/article/details/51607215

'''

import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import acf, pacf


def file_name(file_dir):
    names = []
    for root, dirs, files in os.walk(file_dir):
        #print(root) #当前目录路径
        #print(dirs) #当前路径下所有子目录

        names = files#当前路径下所有非目录子文件

    return names

# 读入文件，并打印出图片
def file_read(name):
    print(name)
    df_data = pd.read_csv('csv/' + name)
    arr_data = df_data.values

    snapId = arr_data[:, 0]
    data = arr_data[:, 1:-5]
    featureName = df_data.columns[1:-5]
    LoadScore = arr_data[:, -4]
    PerScore = arr_data[:, -2]

    return data, LoadScore, PerScore, snapId,featureName




#
# Pearson Correlation

def PC(x, y):

    return  pearsonr(x, y)






if __name__ == '__main__':

    names = file_name('csv/')
    print(names[0])
    x, y1, y2, ID, Fea = file_read(names[0])


    cor = np.zeros([1, len(Fea)])
    for i in range(len(Fea)):

        cor= PC(x[:, i], y2)
        if abs(cor[0]) > 0.128:
            print(Fea[i] + ':')
            print(cor)



















