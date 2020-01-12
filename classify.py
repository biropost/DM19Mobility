import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from scipy import fft

data = pd.read_csv('export.csv', sep=',')
modes = list(data.groupby('mode').count().index[1:])

i = 0

# X: Features
X = None

# y: labels
y = None

# left out 'BICYCLE' (1.8%), 'TRAIN' (1.4%)
for mode in ['BUS', 'CAR', 'METRO', 'TRAM', 'WALK']:
    print("Preprocessing {} data...".format(mode))

    i = i+1

    # get all x,y,z values of this mode
    df_mode = data[data['mode'] == mode][['x', 'y', 'z']]

    # removing stuff that does not fit in an entire sequence with 20 samples
    n = df_mode.shape[0]
    sample_n = 20
    rest = n % sample_n
    df_mode.drop(df_mode.tail(rest).index, inplace=True)
    n = n-rest

    # apply fft to each dimension
    Xx = np.array(df_mode['x']).reshape(int(n / sample_n), sample_n)
    xfft = fft(np.array(df_mode['x']).reshape(int(n / sample_n), sample_n), axis=0)
    Xy = np.array(df_mode['y']).reshape(int(n / sample_n), sample_n)
    yfft = fft(np.array(df_mode['y']).reshape(int(n / sample_n), sample_n), axis=0)
    Xz = np.array(df_mode['z']).reshape(int(n / sample_n), sample_n)
    zfft = fft(np.array(df_mode['z']).reshape(int(n / sample_n), sample_n), axis=0)

    XXX = np.dstack((Xx, Xy, Xz, xfft, yfft, zfft))

    if X is None:
        X = XXX
    else:
        X = np.append(X, XXX, axis=0)

    if y is None:
        y = np.full(shape=(int(n/sample_n)), fill_value=i)
    else:
        y = np.append(y, np.full(shape=(int(n/sample_n)), fill_value=i))

# one-hot-encode true labels
y = to_categorical(y)

print("Starting 5-fold cross-validation...")

model_best = None
score_best = None

groups = data['mode']

kfold = GroupKFold(n_splits=5)

print("X, y is shorter (~1/20) than data['mode'] because FFT was applied to it.")
exit(0)

for train_index, test_index in kfold.split(X, y, groups):
    print("Splitting data in train/test...")
    x_train, x_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    verbose, epochs, batch_size = 0, 10, 32
    samples, features, outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

    print("Building Sequential Model...")
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(samples, features)))
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(outputs, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

    print("Fitting Model to training data...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    print("Evaluating Model...")
    _, accuracy_train = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
    _, accuracy_test = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    print("training accuracy: ", accuracy_train, "testing accuracy: ", accuracy_test)

    if score_best is None:
        score_best = accuracy_test
    elif accuracy_test > score_best:
        score_best = accuracy_test
        model_best = model
