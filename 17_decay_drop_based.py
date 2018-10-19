# Drop-Based Learning Rate Decay
from pandas import read_csv
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import LearningRateScheduler

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

# fix random seed for reproducibility
np.random.seed(7)

# load dataset
filename = 'ionosphere.csv'
dataframe = read_csv(filename, header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:, :34].astype(float)
y = dataset[:, 34]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# create model
model = Sequential()
model.add(Dense(34, input_dim=34, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

# compile model
sgd = SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# fit the model
model.fit(X, y, validation_split=0.33, epochs=50, batch_size=28, callbacks=callbacks_list, verbose=2)