# Multiclass Classification with the Iris Flowers Dataset
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
filename = 'iris.csv'
dataframe = read_csv(filename, header=None)
dataset = dataframe.values
X = dataset[:, :4].astype(float)
y = dataset[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)

# define baseline model
def baseline_model():
    # create
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print('Accuracy: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))