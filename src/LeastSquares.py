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

# Returns the weights using the original naive LS solution
def trainLSOrig(X, y):
    # Find least squares solution with training data provided
    w = np.linalg.inv(X.T@X)@X.T@y
    return w


## Initialize a dataset
num_dataset = 1
data = Dataset(num_dataset) # Test dataset 1

# Use helper functions to get percent error
w = trainLSOrig(data.X_tr, data.y_tr) # Get weights with training set
#print(np.sort(np.abs(w)))
print(w)
y_hat = classify(data.X,w) # Classify test set
perr = get_perr(y_hat, data.y) # Get percent error

# Output results
print("Using original Least Squares")
print("Percent labels misclassified: {}%".format(perr.round(2)))


