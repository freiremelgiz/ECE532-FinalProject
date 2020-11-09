#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


from Dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt

""" Helper Functions """
# Perform data classification with provided weights
def classify(X,w):
    y_hat = np.sign(X@w) # Predict labels
    return y_hat

# Gets the percent error of the predictions
def get_perr(y_hat, y):
    nerr = np.sum(np.abs(y_hat-y)/2) # Number of errors
    perr = nerr/len(y)*100 # Percent error
    return perr

# Returns the weights using the original LS solution
def trainLSOrig(X, y):
    # Find least squares solution with training data provided
    w = np.linalg.inv(X.T@X)@X.T@y
    return w


if __name__ == "__main__":
    # Initialize a dataset
    num_dataset = 1

    data = Dataset(num_dataset) # Retrieve dataset object

    print("-- Using dataset {} --".format(num_dataset))

    # Use helper functions to get percent error
    w = trainLSOrig(data.X_tr, data.y_tr) # Get weights with training set
    y_hat = classify(data.X,w) # Classify test set
    perr = get_perr(y_hat, data.y) # Get percent error

    # Output results
    print("Original Least Squares classification:")
    print("Percent labels misclassified: {}%".format(perr.round(2)))


    """ Low rank approximation study """
    # Find skinny SVD of training data
    U, s, VT = np.linalg.svd(data.X_tr, full_matrices=False)
    V = VT.T

    # Find rank-20 or rank-8 approximation to data
    if num_dataset < 4:
        rank = 20
    elif num_dataset == 6:
        rank = 5
    else:
        rank = 8

    # Compute low-rank approximation to X_tr
    X_r = np.zeros(data.X_tr.shape)
    for i in range(rank):
        X_r += s[i]*np.outer(U[:,i],V[:,i])

    # Use helper functions to get percent error
    w = trainLSOrig(X_r, data.y_tr) # Get weights with training set
    y_hat = classify(data.X,w) # Classify test set
    perr = get_perr(y_hat, data.y) # Get percent error

    # Output results
    print("Least Squares with Rank-{} approximation:".format(rank))
    print("Percent labels misclassified: {}%".format(perr.round(2)))
