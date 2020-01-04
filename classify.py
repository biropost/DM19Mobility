import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from scipy import fft

data = pd.read_csv('export_mobility.csv', sep=',')
data2 = pd.read_csv('export_mobility_v2.csv', sep=',')
data = data[['x', 'y', 'z', 'mode']]
data2 = data2[['x', 'y', 'z', 'mode']]
data = data.append(data2)
data = data.reset_index()
modes = list(data.groupby('mode').count().index[1:])

i = 0
dict = {}
X = None
y = None

for mode in modes:
    if mode in ['BICYCLE', 'BUS', 'CAR', 'METRO', 'TRAM', 'WALK']:
        i = i+1
        df_mode = data[data['mode'] == mode][['x', 'y', 'z']]
        n = df_mode.shape[0]
        rest = n % 10
        df_mode.drop(df_mode.tail(rest).index, inplace=True)
        n = n-rest
        Xx = np.array(df_mode['x']).reshape(int(n / 10), 10)
        xfft = fft(np.array(df_mode['x']).reshape(int(n / 10), 10), axis=0)
        Xy = np.array(df_mode['y']).reshape(int(n / 10), 10)
        yfft = fft(np.array(df_mode['y']).reshape(int(n / 10), 10), axis=0)
        Xz = np.array(df_mode['z']).reshape(int(n / 10), 10)
        zfft = fft(np.array(df_mode['z']).reshape(int(n / 10), 10), axis=0)
        XXX = np.dstack((Xx, Xy, Xz, xfft, yfft, zfft))
        if X is None:
            X = XXX
        else:
            X = np.append(X, XXX, axis=0)
        if y is None:
            y = np.full((int(n/10)), i)
        else:
            y = np.append(y, np.full((int(n/10)), i))
        dict[mode] = i
y = to_categorical(y)


model_best = None
score_best = None

kfold = KFold(5, True, 1)
# enumerate splits
for train, test in kfold.split(X):
    x_train, x_test, y_train, y_test = X[train], X[test], y[train], y[test]
    verbose, epochs, batch_size = 0, 10, 32
    samples, features, outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(samples, features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate the model
    _, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
    if score_best is None:
        score_best = accuracy
    elif accuracy > score_best:
        score_best = accuracy
        model_best = model
    print(accuracy)