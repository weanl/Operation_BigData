

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


t = np.arange(1, 100)

T = 20
PI = 3.14
w = 2*PI/T

S = np.cos(w*t)
R = acf(S, nlags=len(t))
print(len(R))

plot1 = pl.plot(t, R)

pl.show()






































