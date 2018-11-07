# Load and Plot the IMDB dataset
import numpy as np
from keras.datasets import imdb
from matplotlib import pyplot

# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# summarize size
print('Training data: ')
print(X.shape)
print(y.shape)

# summarize number of classes
print('Classes: ')
print(np.unique(y))

# summarize number of words
print('Number of words: ')
print(len(np.unique(np.hstack(X))))

# summarize review length
print('Review length: ')
result = [len(x) for x in X]
print('Mean %.2f words (%f)' % (np.mean(result), np.std(result)))

# plot review length as a boxplot and histogram
pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()