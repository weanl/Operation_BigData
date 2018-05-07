import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.cross_validation import train_test_split
from awr53522DataSet import LoadData



def runRegression(X_train, X_test, y_train, y_test, method='LR'):
    print('Regression Method is ' + method + '-----------------------')
    if method == 'LR':
        regr = linear_model.LinearRegression()
    if method == 'BRR':
        regr = linear_model.BayesianRidge()
    if method == 'DTR':
        regr = tree.DecisionTreeRegressor(max_depth=6)
    if method == 'GBR':
        regr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
        max_depth=2, random_state=0, loss='ls')
    if method == 'MLPR':
        regr = MLPRegressor(hidden_layer_sizes=(41,), batch_size=2048, max_iter=2000000, learning_rate_init=0.001)

    print(regr)
    regr.fit(X_train, y_train)

    y_pred = regr.predict(X_train)
    MSE_train = mean_squared_error(y_train, y_pred)

    y_pred = regr.predict(X_test)
    MSE_pred = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test, y_pred)

    print("MSE_train: %f", MSE_train)
    print(" MSE_pred: %f", MSE_pred)
    print("       R2: %f", R2)

    return MSE_train, MSE_pred, R2

'''
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
'''






if __name__ == '__main__':

    X, y = LoadData(target='PerfScore')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    runRegression(X_train, X_test, y_train, y_test, method='LR')
    runRegression(X_train, X_test, y_train, y_test, method='BRR')
    runRegression(X_train, X_test, y_train, y_test, method='DTR')
    runRegression(X_train, X_test, y_train, y_test, method='GBR')
    runRegression(X_train, X_test, y_train, y_test, method='MLPR')











'''
End Of File
'''
