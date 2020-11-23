#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


from Dataset import Dataset
from LeastSquares import get_perr
from LeastSquares import classify
import numpy as np
import matplotlib.pyplot as plt

""" Helper Functions """
# Computes the gradient of the Hinge Loss cost function
## X - data matrix with datapoints as rows
## y - label vector
## w - current GD step weights
def get_HL_gradient(X, y, w):
    grad = 0 # initialize to 0
    for i in range(X.shape[0]):
        grad += 0.5*y[i]*(1 + np.sign(1 - y[i]*np.dot(X[i,:],w)))*X[i,:]
    grad = -grad # Negate
    return grad

if __name__ == "__main__":
    # Initialize a dataset
    num_dataset = 1
    data = Dataset(num_dataset) # Retrieve dataset object
    print("-- Using dataset {} --".format(num_dataset))

    # Use Gradient Descent to converge to the Hinge Loss solution
    tau = 0.9/(np.linalg.norm(data.X_tr, 2)**2) # Ensure step size is in safe range
    # Initialize w
    w = np.zeros(data.cols)
    # Iterate to converge
    for k in range(10): # 20 times
        grad = get_HL_gradient(data.X_tr, data.y_tr, w)
#        print("Gradient: {}".format(grad))
        w = w - tau*grad
        print("Norm of w: {}".format(np.linalg.norm(w,2)))

    # Review performance
    y_hat = classify(data.X,w)
    perr = get_perr(y_hat,data.y)

    # Output results
    print("Percent labels misclassified: {}%".format(perr.round(2)))
