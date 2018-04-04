import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.cross_validation import train_test_split

file_sets = [
        'DBID(1002089510)_INSTID(1)','DBID(2897570545)_INSTID(1)',
        'DBID(1227435885)_INSTID(1)','DBID(2949199900)_INSTID(1)',
        'DBID(1227435885)_INSTID(2)','DBID(3065831173)_INSTID(1)',
        'DBID(1254139675)_INSTID(1)','DBID(3111200895)_INSTID(1)',
        'DBID(1384807946)_INSTID(1)','DBID(3172835364)_INSTID(1)',
        'DBID(1624869053)_INSTID(1)','DBID(3204204681)_INSTID(1)',
        'DBID(1636599671)_INSTID(1)','DBID(3482311182)_INSTID(1)',
        'DBID(1636599671)_INSTID(2)','DBID(349165204)_INSTID(1)',
        'DBID(172908691)_INSTID(1)','DBID(3671658776)_INSTID(1)',
        'DBID(1855232979)_INSTID(1)','DBID(3671658776)_INSTID(2)',
        'DBID(1982696497)_INSTID(1)','DBID(3775482706)_INSTID(1)',
        'DBID(2031853600)_INSTID(1)','DBID(3775482706)_INSTID(2)',
        'DBID(2052255707)_INSTID(1)','DBID(4213264717)_INSTID(1)',
        'DBID(2238741707)_INSTID(1)','DBID(4215505906)_INSTID(1)',
        'DBID(2238741707)_INSTID(2)','DBID(4225426100)_INSTID(1)',
        'DBID(2328880794)_INSTID(1)','DBID(4291669003)_INSTID(1)',
        'DBID(2413621137)_INSTID(1)','DBID(4291669003)_INSTID(2)',
        'DBID(2612437783)_INSTID(1)','DBID(447326245)_INSTID(1)',
        'DBID(2644427317)_INSTID(1)','DBID(468957624)_INSTID(1)',
        'DBID(2707003786)_INSTID(1)','DBID(505574722)_INSTID(1)',
        'DBID(2762567375)_INSTID(1)','DBID(522516877)_INSTID(1)',
        'DBID(2768077198)_INSTID(1)','DBID(770699067)_INSTID(1)',
        'DBID(2778659381)_INSTID(1)','DBID(929227073)_INSTID(1)',
        'DBID(2778659381)_INSTID(2)','DBID(942093433)_INSTID(1)',
        'DBID(2802676787)_INSTID(1)','DBID(998852395)_INSTID(1)'
        ]






def linearRegression(X_train, y_train, X_test, y_test):
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_train)
    MSE_train = mean_squared_error(y_train, y_pred)

    y_pred = regr.predict(X_test)
    MSE_pred = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)

    return MSE_train, MSE_pred, R2

def runlinearRegression(PATH, DBID):

    data = pd.read_csv(PATH + DBID + '.csv')


    values = data.values
    X = values[:, 2:-5]
    y = values[:, -4]
    data_len = y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    MSE_train, MSE_pred, R2 = linearRegression(X_train, y_train, X_test, y_test)
    print("MSE_train: %f", MSE_train)
    print(" MSE_pred: %f", MSE_pred)
    print("       R2: %f", R2)

    return data_len, MSE_train, MSE_pred, R2


def sprt_run():
        for i in range(len(file_sets)):
            DBID = file_sets[i]
            data_len, MSE_train, MSE_pred, R2 = runlinearRegression('../FS/csv_Preprocess1.0/', DBID)
            if i == 0:
                sum_len = data_len
                sum_MSE_train = MSE_train * data_len * 0.8
                sum_MSE_pred = MSE_pred * data_len * 0.2
            else:
                sum_len = sum_len + data_len
                sum_MSE_train = sum_MSE_train + MSE_train * data_len * 0.8
                sum_MSE_pred = sum_MSE_pred + MSE_pred * data_len * 0.2

        print("total samples: ", sum_len)
        print("avrg_MSE_train: ", sum_MSE_train/(sum_len * 0.8))
        print("avrg_MSE_pred: ", sum_MSE_pred/(sum_len * 0.2))

def all_run():
    DBID = 'all_data'
    data_len, MSE_train, MSE_pred, R2 = runlinearRegression('../FS/csv_Preprocess2.0/', DBID)

if __name__ == '__main__':

    sprt_run()
