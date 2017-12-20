
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import os
import DataStream as DS

PATH = '../FS/csv/'

# 读入文件，并打印出图片
def file_csv(name):
    print(name)
    df_data = pd.read_csv('csv/' + name)
    arr_data = df_data.values

    snapID = arr_data[:, 0]
    performSore = arr_data[:, 2]

    fig, ax = plt.subplots()

    ax.plot(snapID, performSore)
    ax.set(xlabel='snapID', ylabel='0~100',
       title='the certain feature')
    ax.grid()

    #fig.savefig(name + ".png")

    plt.show()





if __name__ == '__main__':
    print("Here are the result:")

    #FileNames = DS.fetch_FileNames(PATH)
    #name = FileNames[0]
    #data, LoadScore, PerfScore, snapId, featureName = DS.DataPrepared(name)


    FEATRUE = '2080031'
    SCORE_NAME = 'LoadScore'
    fileName = 'DBID(2031853600)_INSTID(1).csv'
    df_DBID = pd.read_csv(PATH + fileName)
    # arr_data = df_DBID.values
    # LoadScore = arr_data[:, -4]
    # print(type(LoadScore))
    # DS.DataNormalized(LoadScore)
    # print(LoadScore)



    y1 = df_DBID[FEATRUE].values
    y2 = df_DBID[SCORE_NAME].values
    x =  df_DBID['SnapId'].values

    print(type(y2))
    #DS.DataNULLProcess(x1)
    DS.DataNormalized(y1)
    #DS.DataNULLProcess(x2)
    DS.DataNormalized(y2)

    print(y1)
    print(y2)

    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    y = np.array([11,12,13,14], dtype=np.float)
    print(y.shape)
    DS.DataNormalized(y)
    print(y.shape)
    print(y)

    plot1, = pl.plot(x, y1, 'r')
    plot2, = pl.plot(x, y2, 'g')

    pl.xlabel(fileName + ':' + 'snapId')
    pl.ylabel('normalized value')

    pl.legend([plot1, plot2], (FEATRUE, SCORE_NAME))
    pl.show()






























