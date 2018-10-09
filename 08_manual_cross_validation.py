# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
filename = 'pima-indians-diabetes.csv'
dataset = np.loadtxt(filename, delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, :8]
y = dataset[:, 8]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the model
    model.fit(X[train], y[train], epochs=150, batch_size=10, verbose=0)

    # evaluate the model
    scores = model.evaluate(X[test], y[test], verbose=0)
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))