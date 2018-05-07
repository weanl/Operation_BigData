
import numpy as np
import pandas as pd




def LoadData(target='LoadScore'):
    data = pd.read_csv('../FS/csv_Preprocess2.0/all_data.csv')
    values = data.values
    X = values[:, 3:-4]
    if target == 'LoadScore':
        y = values[:, -4]
    elif target == 'PerfScore':
        y = values[:, -2]
    elif target == 'LoadLevel':
        y = values[:, -3]
    elif target == 'PerfLevel':
        y = values[:, -1]

    return X, y











if __name__ == '__main__':

    print('This is a python script for awr data interface!!!')
