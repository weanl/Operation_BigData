

'''

Feature Selection

adopt a few methods
from Pair-wise to accumulated

reference:
    http://blog.csdn.net/bryan__/article/details/51607215
    https://zhuanlan.zhihu.com/ai-insight

'''



import os
import pandas as pd
import numpy as np
import math


PATH = '../FS/csv/'


# 遍历文件名
def fetch_FileNames(file_dir):
    FileNames = []
    for root, dirs, files in os.walk(file_dir):
        #print(root) #当前目录路径
        #print(dirs) #当前路径下所有子目录

        FileNames = files#当前路径下所有非目录子文件

    return FileNames

# 依据文件名对应文件（并非顺序）
# 并对 data的奇异值做 均值处理
# null --> type(nan)=float 而type(None)=NoneType
# 这里调用 math.isnan() 好像不可以用 float("nan") 做比较
def read_eachFile(name):

    df_data = pd.read_csv(PATH + name)
    arr_data = df_data.values

    snapId = arr_data[:, 0]
    data = arr_data[:, 1:-5]


    featureName = df_data.columns[1:-5]
    LoadScore = arr_data[:, -4]
    PerScore = arr_data[:, -2]

    return data, LoadScore, PerScore, snapId,featureName

def DataNULLProcess(M):
    row, col = M.shape

    # 缺失值处理
    for j in range(col):
        for i in range(row):
            if math.isnan(M[i, j]):
                # 第j列有缺失
                print(j)

                flag = False
                sum = 0
                count = 0

                # 尝试求出第j列的均值
                for i in range(row):
                    if math.isnan(M[i, j]):
                        continue
                    else:
                        flag = True
                        sum = sum + M[i, j]
                        count += 1
                # 如果求出了均值
                if flag:
                    mean = sum / count
                    for i in range(row):
                        if math.isnan(M[i, j]):
                            M[i, j] = mean
                # 第j列全部值都是缺失的，补另零
                else:
                    for i in range(row):
                        M[i, j] = 0
                break

# 对一维矩阵进行归一化处理 到(0,1)区间上
def DataNormalized(arr):
    Max = arr.max()
    Min = arr.min()
    L = Max - Min
    length = len(arr)
    if L != 0:
        for i in range(length):
            arr[i] = (arr[i] - Min) / L

def DataPrepared(name):

    data, LoadScore, PerScore, snapId, featureName = read_eachFile(name)
    DataNULLProcess(data)
    row, col = data.shape
    for j in range(col):
        DataNormalized(data[:, j])

    DataNormalized(LoadScore)
    DataNormalized(PerScore)

    return data, LoadScore, PerScore, snapId, featureName






if __name__ == "__main__":

    FileNames = fetch_FileNames(PATH)
    FileNames.sort() # 并非实际上大额顺序，只是按照字符串的大小比较
    print("There are totally " + str(len(FileNames)) + " csv files.")
    

















