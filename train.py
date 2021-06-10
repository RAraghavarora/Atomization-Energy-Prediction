from matplotlib import pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from kerastuner.tuners import Hyperband

from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import pdb

import scipy
import scipy.io

import tensorflow as tf
from tensorflow import keras

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


# Start the learning rate from 10^(-4) and going on till 10^(-6)
# lr = lr0/(1+kt) where lr, k are hyperparameters and t is the iteration number
lr0 = 10**(-4)
epochs = 20000
decay_rate = 99 / epochs 
momentum = 0.8
sgd = keras.optimizers.SGD(learning_rate=lr0, momentum=momentum, decay=decay_rate, nesterov=False)


def build_model(hp):
    # Define the NN model
    model = Sequential()

    # Add an input layer 
    model.add(keras.layers.Flatten(input_shape=(529,)))

    # Add one hidden layer with nodes between 32 and 1024
    hp_units = hp.Int('unit1', min_value=32, max_value=1024, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='tanh'))

    # Add one hidden layer with nodes between 32 and 1024
    hp_units = hp.Int('unit2', min_value=32, max_value=1024, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='tanh'))

    # Add an output layer 
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae'])

    return model


# Define a tuner to tune the hyperparameters
tuner = Hyperband(
    build_model,
    objective='val_mae',
    max_epochs=20,
    factor=3,
    directory='tuner_data/',
    project_name='atomization_energy_prediction')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=350)

# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=jobs, inter_op_parallelism_threads=jobs)))


# fitted = model.fit(X_train, y_train, epochs=epochs, batch_size=mb, verbose=1)

tuner.search(X_train, y_train, epochs=epochs, validation_split=0.33, batch_size=mb, verbose=1, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
pdb.set_trace()

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('unit1')} and the 2nd layer is {best_hps.get('unit2')}
""")

# Build the model
model = tuner.hypermodel.build(best_hps)
fitted = model.fit(X_train, y_train, epochs=epochs, validation_split=0.33, batch_size=mb, verbose=1)

pdb.set_trace()

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
plt.savefig('kt_AE2.png', dpi=600)
plt.close()

plt.plot(fitted.history['mae'])
plt.xlabel('Epochs')
plt.ylabel('MAE (kcal/mol)')
plt.savefig('kt_Keras_Results.png', dpi=600)
plt.close()

f1 = open("keras_output.txt", "a")
for test, true in zip(predicted, y_test):
    f1.write('%d,%d\n' % (true, test))

pdb.set_trace()
