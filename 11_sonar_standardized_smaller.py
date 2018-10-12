# Binary Classification with Sonar Dataset: Standardized Smaller
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
filename = 'sonar.csv'
dataframe = read_csv(filename, header=None)
dataset = dataframe.values

# split into input (X) and output (y) variables
X = dataset[:, :60].astype(float)
y = dataset[:, 60]

# encode class valus as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# smaller model
def create_smaller():
    # create
    model = Sequential()
    model.add(Dense(30, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_y, cv=kfold)
print('Smaller: %.2f%% (%.2f%%)' % (results.mean() * 100, results.std() * 100))