# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
np.random.seed(7)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# load weights
model.load_weights('weights.best.hdf5')

# compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Created model and loaded weights from file')

# load pima indians dataset
filename = 'pima-indians-diabetes.csv'
dataset = np.loadtxt(filename, delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, :8]
y = dataset[:, 8]

# estimate accuracy on whole dataset using loaded weights
scores = model.evaluate(X, y, verbose=0)
print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))