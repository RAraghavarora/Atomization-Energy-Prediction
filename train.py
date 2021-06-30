from matplotlib import pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from kerastuner.tuners import BayesianOptimization

from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pdb

import scipy
import scipy.io

import tensorflow as tf
from tensorflow import keras

jobs = 7  # it means number of cores
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

mb = 32  # Size of the minibatch
split = 5
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
    # As the matrix is symmetric, we take only the lower diagonal
    x = x[np.triu_indices_from(x)]
    pdb.set_trace()
    return x


X_train = np.array([process_matrix(train_example) for train_example in X_train])
X_test = np.array([process_matrix(test_example) for test_example in X_test])

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


def build_model(hp):
    # Define the NN model
    model = Sequential()

    # Add an input layer 
    model.add(keras.layers.Flatten(input_shape=(276,)))

    # Add one hidden layer with nodes between 32 and 256
    hp_units = hp.Int('unit1', min_value=32, max_value=256, step=32)
    hp_activation = hp.Choice('activation',
                              values=['relu', 'tanh', 'sigmoid'],
                              default='relu'
                              )
    model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))

    # Add one hidden layer with nodes between 32 and 256
    hp_units = hp.Int('unit2', min_value=32, max_value=256, step=32)

    model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))

    # Add one hidden layer with nodes between 32 and 256
    hp_units = hp.Int('unit3', min_value=32, max_value=256, step=32)  # Step of 32

    model.add(keras.layers.Dense(units=hp_units, activation=hp_activation))

    # Add an output layer 
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae'])

    return model


class MyTuner(BayesianOptimization):
    def run_trial(self, trial, *args, **kwargs):
        '''
        Overriding the run_trial method to change the batchsize and the number of epochs
        '''
        kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 64, step=32)
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', 20, 100)
        super(MyTuner, self).run_trial(trial, *args, **kwargs)


# Define a tuner to tune the hyperparameters
tuner = MyTuner(
    build_model,
    objective='val_mae',
    max_trials=500,
    project_name='AE_prediction_final',
    directory='/Volumes/Seagate Por/Thesis/bo/')

# If the mae does not improve over a span of 30 epochs, stop the training.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=30)

tuner.search(X_train, y_train, validation_split=0.33, shuffle=True, use_multiprocessing=True, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

pdb.set_trace()

# Build the model
model = tuner.hypermodel.build(best_hps)

# If the val_mae does not change by at least 2 units over a span of 100 epochs, stop the training.
stop_early = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=500)

# Change the learning rate after each epoch based on the relation: lr=lr0/(1+kt)


def adapt_learning_rate(epoch):
    lr0 = 10**(-4)
    epochs = 20000
    decay_rate = 99 / epochs 
    lr = lr0 / (1 + decay_rate * epoch)
    return lr


epochs = 20000
my_lr_scheduler = keras.callbacks.LearningRateScheduler(adapt_learning_rate)


# Train the model
fitted = model.fit(X_train, y_train, epochs=epochs, shuffle=True, use_multiprocessing=True, batch_size=mb, verbose=1, callbacks=[stop_early, my_lr_scheduler])

pdb.set_trace()

# Test the model
predicted = model.predict(X_test)

print(f"""
The hyperparameter search is complete. 
{best_hps.get('unit1')} --- {best_hps.get('activation1')} 
{best_hps.get('unit2')} --- {best_hps.get('activation2')} 
{best_hps.get('unit3')} --- {best_hps.get('activation3')} 
""")
print('SMAE on the test set is {:}'.format(mean_squared_error(predicted, y_test)))
print('MAE on the test set is {:}'.format(mean_absolute_error(predicted, y_test)))
print('STD of the property is {:}'.format(y_test.std()))

# Plot the results
x = range(len(y_test))
plt.scatter(x, y_test, marker='o', c='blue', label='exact')
plt.scatter(x, predicted, marker='+', c='red', label='predicted')
plt.legend(scatterpoints=1)
plt.savefig('kt_AE2_3.png', dpi=600)
plt.close()

plt.plot(fitted.history['mae'])
plt.xlabel('Epochs')
plt.ylabel('MAE (kcal/mol)')
plt.savefig('kt_Keras_Results_3.png', dpi=600)
plt.close()

f1 = open("keras_output_3.txt", "a")
for test, true in zip(predicted, y_test):
    f1.write('%d,%d\n' % (true, test))

pdb.set_trace()
