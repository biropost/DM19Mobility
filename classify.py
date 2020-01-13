import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
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
print("Modes:", modes)

i = 0

# X: Features
X = None

# y: labels
y = None

# 20 seconds per sample
sample_n = 20

# left out 'BICYCLE' (1.8%) and 'TRAIN' (1.4%)
for mode in ['BUS', 'CAR', 'METRO', 'TRAM', 'WALK']:
    print("Preprocessing {} data...".format(mode))

    # mode id (needed for one-hot-encoding afterwards)
    i = i+1

    # get all rows of current mode
    df_mode = data[data['mode'] == mode]

    # per-trip pre-processing is necessary because we need to keep the data-tripId-relationship for the grouped k-fold.
    trip_ids = list(df_mode.groupby('trip_id').count().index[1:])
    for trip_id in trip_ids:
        df_mode_trip = df_mode[df_mode['trip_id'] == trip_id]

        # removing stuff that does not fit in an entire sequence with 20 samples
        n = df_mode_trip.shape[0]  # n = 7819 rows
        rest = n % sample_n  # rest = 19
        df_mode_trip = df_mode_trip.drop(df_mode_trip.tail(rest).index)  # removing the last 19 elements
        n = n - rest  # n is now 7800
        new_n = int(n / sample_n)  # new_n = 390

        # reshape to 20 columns and apply fft
        Xx = np.array(df_mode_trip['x']).reshape(new_n, sample_n)
        xfft = fft(Xx, axis=0)
        Xy = np.array(df_mode_trip['y']).reshape(new_n, sample_n)
        yfft = fft(Xy, axis=0)
        Xz = np.array(df_mode_trip['z']).reshape(new_n, sample_n)
        zfft = fft(Xz, axis=0)

        # here is where my mind gave up.
        # need to pass the trip_id along with the data in XXX.
        # not sure how, because XXX.shape is (something, 20, 6) - i.e. 3-dimensional.

        XXX = np.dstack((Xx, Xy, Xz, xfft, yfft, zfft))

        if X is None:
            X = XXX
        else:
            X = np.append(X, XXX, axis=0)

        if y is None:
            y = np.full(shape=new_n, fill_value=i)
        else:
            y = np.append(y, np.full(shape=new_n, fill_value=i))

# one-hot-encode true labels
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting 5-fold cross-validation...")
print("Cannot do Cross-Validation: X, y is shorter (~1/20) than data['trip_id'] because FFT was applied to it.")
exit(0)

model_best = None
score_best = None

# use the id of the per-trip-fft-data here for a correct grouped k-fold split.
groups = data['trip_id']

kfold = GroupKFold(n_splits=5)

for train_index, test_index in kfold.split(X_train, y_train, groups):
    print("Splitting data in train/test...")
    x_train_fold, x_test_fold, y_train_fold, y_test_fold = X_train[train_index], X_train[test_index], y_train[train_index], y_train[test_index]
    verbose, epochs, batch_size = 0, 10, 32
    samples, features, outputs = x_train_fold.shape[1], x_train_fold.shape[2], y_train_fold.shape[1]

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
    model.fit(x_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, verbose=verbose)

    print("Evaluating Model...")
    _, accuracy_train = model.evaluate(x_train_fold, y_train_fold, batch_size=batch_size, verbose=0)
    _, accuracy_test = model.evaluate(x_test_fold, y_test_fold, batch_size=batch_size, verbose=0)

    print("training accuracy: ", accuracy_train, "testing accuracy: ", accuracy_test)

    if score_best is None:
        score_best = accuracy_test
    elif accuracy_test > score_best:
        score_best = accuracy_test
        model_best = model

# Evaluation

_, accuracy = model_best.evaluate(X_test, y_test)

print("Accuracy:", accuracy)

y_pred = model_best.predict(X_test)

print(precision_recall_fscore_support(y_test, y_pred))
