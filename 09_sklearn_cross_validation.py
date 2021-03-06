# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

# function to create model, required for KerasClassifier
def create_model():
    # create
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
filename = 'pima-indians-diabetes.csv'
dataset = np.loadtxt(filename, delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:, :8]
y = dataset[:, 8]

# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())