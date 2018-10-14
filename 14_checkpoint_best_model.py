# Checkpoint the weights for best model on validation accuracy
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
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
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint
filepath = 'weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, 
    mode='max')
callbacks_list = [checkpoint]

# fit the model
model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, 
    verbose=0)