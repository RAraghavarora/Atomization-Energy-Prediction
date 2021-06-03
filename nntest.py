import os
import pickle
import sys
import numpy
import copy
import scipy
import scipy.io

# --------------------------------------------
# Parameters
# --------------------------------------------
split = int(3)  # test split for cross-validation (between 0 and 5)

# --------------------------------------------
# Load data and models
# --------------------------------------------
# if not os.path.exists('./Dataset/qm7.mat'):
#     os.system('wget http://www.quantum-machine.org/data/qm7.mat')
dataset = scipy.io.loadmat('./Dataset/qm7.mat')
# nn = pickle.load(open('nn-%d.pkl' % split, 'r'))
nn = pickle.load(open('nn.pkl', 'rb'))

print('results after %d iterations' % nn.nbiter)

temp = [*range(0, split), *range(split + 1, 5)]
Ptrain = dataset['P'][temp].flatten()
Ptest = dataset['P'][split]

f1 = open("output.txt", "a")

for P, name in zip([Ptrain, Ptest], ['training', 'test']):
    # --------------------------------------------
    # Extract test data
    # --------------------------------------------
    X = dataset['X'][P]
    T = dataset['T'][0, P]

    # --------------------------------------------
    # Test the neural network
    # --------------------------------------------
    print('\n%s set:' % name)
    Y = numpy.array([nn.forward(X) for _ in range(10)]).mean(axis=0)
    print('MAE:  %5.2f kcal/mol' % numpy.abs(Y - T).mean(axis=0))
    print('RMSE: %5.2f kcal/mol' % numpy.square(Y - T).mean(axis=0)**.5)
    f1.write("**********\n")
    for true, test in zip(T, Y):
        f1.write('%d,%d\n' % (true, test))

f1.close()
