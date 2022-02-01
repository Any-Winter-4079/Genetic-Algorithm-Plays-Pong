#################
#####  NET  #####
#################

import numpy as np
from config import *


def forward(input_features, genes, debug=False):
    """Transform input features observed from the world [color, position, e.g.] into
       a decision output [action, e.g. move up, or opinion, e.g. this is a cat]
       -> input_features is an np_array
       -> genes is an np_array
    """
    W1 = genes[:n_input_features *
               n_hidden_units].reshape((n_input_features, n_hidden_units))
    z2 = np.dot(input_features, W1)
    a2 = sigmoid(z2)
    W2 = genes[n_input_features *
               n_hidden_units:].reshape((n_hidden_units, n_output_features))
    z3 = np.dot(a2, W2)
    # yHat = sigmoid(z3)
    # Classes must be mutually exclusive (use Logistic Regression otherwise)
    yHat = softmax(z3)
    if debug:
        print("Input features X: {}".format(input_features))
        print("W1: {}".format(W1))
        print("z2: {}".format(z2))
        print("a2: {}".format(a2))
        print("W2: {}".format(W2))
        print("z3: {}".format(z3))
        print("yHat: {}".format(yHat))
    return yHat


def sigmoid(z):
    """Apply sigmoid activation function"""
    return 1/(1 + np.exp(-z))


def softmax(yHat):
    """Apply softmax activation function"""
    return np.exp(yHat) / np.sum(np.exp(yHat), axis=0)
