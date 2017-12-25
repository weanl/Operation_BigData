

'''
TRY FILES FOR ACF and ARIMA

created by weanl

refer to:
    https://www.cnblogs.com/bradleon/p/6832867.html


'''




from statsmodels.tsa.stattools import acf, pacf
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


def ACF(S, LEN):

    return acf(S, nlags=LEN)

def CycleLeftShift(S, n):

    L = len(S)
    if n > L:
        m,n = divmod(n, L)
        print("converted to "+str(n))
    temp = np.zeros(n)
    for i in range(n):
        temp[i] = S[i]

    for i in range(L-n):
        S[i] = S[i+n]

    for i in range(n):
        S[L-n+i] = temp[i]



def myACF(S, LEN):
    R = np.zeros(LEN)
    SS = np.zeros(LEN)
    for i in range(LEN):
        SS[i] = S[i]

    for n in range(LEN):
        CycleLeftShift(SS, n)
        R[n] = S.dot(SS)

    return R






if __name__ == "__main__":
    t = np.arange(0, 100)
    x = np.ones(100)
    T = 20
    PI = 3.14
    w = 2*PI/T

    S = np.cos(w*t)
    R = myACF(x, len(t))
    print(len(R))

    plot1 = pl.plot(t, x)

    pl.show()

    xx = np.arange(1,10)
    print(xx)
    CycleLeftShift(xx, 23)
    print(xx)





































