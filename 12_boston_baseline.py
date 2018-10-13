# Regression Example With Boston Dataset: Baseline
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
filename = 'housing.csv'
dataframe = read_csv(filename, delim_whitespace=True, header=None)
dataset = dataframe.values

# split into input (X) and output (y) variables
X = dataset[:, :13]
y = dataset[:, 13]

# define base model
def baseline_model():
    # create
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    # compile
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print('Baseline: %.2f (%.2f) MSE' % (results.mean(), results.std()))