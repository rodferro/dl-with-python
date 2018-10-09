# MLP with automatic validation set
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

# load pima indians dataset
filename = 'pima-indians-diabetes.csv'
dataset = np.loadtxt(filename, delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, :8]
y = dataset[:, 8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10)