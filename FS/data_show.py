
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def file_name(file_dir):
    names = []
    for root, dirs, files in os.walk(file_dir):
        #print(root) #当前目录路径
        #print(dirs) #当前路径下所有子目录

        names = files#当前路径下所有非目录子文件

    return names

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

def data_check(name):
    print(name)
    df_data = pd.read_csv('csv/' + name)
    arr_data = df_data.values

    print(arr_data[0][0])
    for i in range(arr_data.shape[0]):
        if i==0:
            continue
        temp = arr_data[i][0]-arr_data[i-1][0]
        if temp != 1:
            return arr_data[i][0]



if __name__ == '__main__':
    print("Here are the result:")
    names = file_name('csv/')

    #for name in names:
        #file_csv(name)

    file_csv(names[0])
































