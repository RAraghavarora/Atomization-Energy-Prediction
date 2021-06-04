from matplotlib import pyplot as plt
import pdb

from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import scipy
import scipy.io
from keras import backend as K
import tensorflow as tf
jobs = 7  # it means number of cores
tf.config.threading.set_intra_op_parallelism_threads(7)
tf.config.threading.set_inter_op_parallelism_threads(7)

mb = 32  # Size of the minibatch
split = 4
# split = int(sys.argv[1])  # test split

dataset = scipy.io.loadmat('./Dataset/qm7.mat')  # Using QM7 dataset

# Extracting the data
temp = [*range(0, split), *range(split + 1, 5)]
P = dataset['P'][temp].flatten()
X = dataset['X'][P]  # Input
T = dataset['T'][0, P]  # Output

# Separate train and test data
X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.33, random_state=42)
input_size = X_train.shape[0]  # Total training data size

# Preprocessing
noise = 1.0


def process_matrix(x):
    '''
    Process the Coulomb matrix input. Generate randomly sorted coulomb matrix by adding random noise to each row.
    Flatten the matrix to make it 1D
    '''
    inds = np.argsort(-(x**2).sum(axis=0)**.5 + np.random.normal(0, noise, x[0].shape))
    try:
        x = x[inds, :][:, inds] * 1
    except:
        print(x)
        quit()
    x = x.flatten()
    return x


X_train = np.array([process_matrix(train_example) for train_example in X_train])
X_test = np.array([process_matrix(test_example) for test_example in X_test])

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Define the NN model
model = Sequential()

# Add an input layer 
model.add(Dense(400, activation='tanh', input_shape=(529,)))

# Add one hidden layer 
model.add(Dense(100, activation='tanh'))

# Add an output layer 
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])

# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=jobs, inter_op_parallelism_threads=jobs)))

fitted = model.fit(X_train, y_train, epochs=20000, batch_size=mb, verbose=1)


# Test the model
predicted = model.predict(X_test)
print('SMAE on the test set is {:}'.format(mean_squared_error(predicted, y_test)))
print('MAE on the test set is {:}'.format(mean_absolute_error(predicted, y_test)))
print('STD of the property is {:}'.format(y_test.std()))

# Plot the results
x = range(len(y_test))
plt.scatter(x, y_test, marker='o', c='blue', label='exact')
plt.scatter(x, predicted, marker='+', c='red', label='predicted')
plt.legend(scatterpoints=1)
plt.savefig('AE2.png', dpi=600)
plt.close()

plt.plot(fitted.history['mae'])
plt.xlabel('Epochs')
plt.ylabel('MAE (kcal/mol)')
plt.savefig('Keras_Results.png', dpi=600)
plt.close()

f1 = open("keras_output.txt", "a")
for test, true in zip(predicted, y_test):
    f1.write('%d,%d\n' % (true, test))

pdb.set_trace()
