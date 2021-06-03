import os
import pickle
import sys
import numpy
import nn
import copy
import scipy
import scipy.io

# --------------------------------------------
# Parameters
# --------------------------------------------
seed = 3453
split = int(sys.argv[1])  # test split

mb = 25     # size of the minibatch
hist = 0.1    # fraction of the history to be remembered

# --------------------------------------------
# Load data
# --------------------------------------------
numpy.random.seed(seed)
dataset = scipy.io.loadmat('./Dataset/qm7.mat')  # Using QM7 dataset

# --------------------------------------------
# Extract training data
# --------------------------------------------
temp = [*range(0, split), *range(split + 1, 5)]
P = dataset['P'][temp].flatten()
X = dataset['X'][P]
T = dataset['T'][0, P]

# --------------------------------------------
# Create a neural network
# --------------------------------------------
I, O = nn.Input(X), nn.Output(T)
print('hello')
nnsgd = nn.Sequential([I, nn.Linear(I.nbout, 400), nn.Sigmoid(), nn.Linear(400, 100), nn.Sigmoid(), nn.Linear(100, O.nbinp), O])
nnsgd.modules[-2].W *= 0
nnavg = copy.deepcopy(nnsgd)

# --------------------------------------------
# Train the neural network
# --------------------------------------------
for i in range(1, 1000001):

    if i > 0:
        lr = 0.001  # learning rate
    if i > 500:
        lr = 0.0025
    if i > 2500:
        lr = 0.005  # Change this!
    if i > 12500:
        lr = 0.01

    r = numpy.random.randint(0, len(X), [mb])  # All the features of random 25 training examples. (mb = 25)
    Y = nnsgd.forward(X[r])
    nnsgd.backward(Y - T[r])
    nnsgd.update(lr)
    nnavg.average(nnsgd, (1 / hist) / ((1 / hist) + i))
    nnavg.nbiter = i

    if i % 100 == 0:
        pickle.dump(nnavg, open('nn-%d.pkl' % split, 'wb'), pickle.HIGHEST_PROTOCOL)
