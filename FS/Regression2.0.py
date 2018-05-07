
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from awr53522DataSet import LoadData
from sklearn.cross_validation import train_test_split

from ann_visualizer.visualize import ann_viz







if __name__ == '__main__':

    X, y = LoadData(target='LoadScore')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential()
    model.add(Dense(input_dim=41, units=1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')

    history = model.fit(X_train, y_train, batch_size=2048, nb_epoch=3000)

'''
    print('Start Training -----------------------')
    for step in range(80001):
        history = model.fit(X_train, y_train, batch_size=128)
        loss = history.history['loss'][0]
        if step % 50 == 0:
            print("After %d trainings, the cost: %f" % (step, loss))


    print('Start Testing -----------------------')
    loss = model.evaluate(X_test, y_test, batch_size=40)
    print('test cost:', loss)

'''
# ann_viz(model)
