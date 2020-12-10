#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"

"""
This script solves the Support Vector Machine
classification problem to train a set of linear
weights.

"""

import numpy as np
from IterReg import IterReg
from Dataset import Dataset
from HingeLoss import get_loss_HL
from LeastSquares import classify
from LeastSquares import get_perr

DATASET = 1
ALGO = 'GDSVM'

LAMB_ARR = (1e-6,19.95,19.95,1.6e-5,1.4e-6,19.95) # From CV Ridge Reg

""" Helper Functions """
get_loss_SVM = get_loss_HL # Same cost function for SVM and HL

# Take a step in GD
def step_GDSVM(X, y, w, lamb, tau):
    grad = 0 # Init to 0
    for i in range(X.shape[0]):
        grad -= 0.5*y[i]*(1 + np.sign(1 - y[i]*np.dot(X[i,:],w)))*X[i,:]
    grad += 2*lamb*w # Regularize
    w_new = w - tau*grad
    return w_new


if __name__ == "__main__":
    # Initialize a dataset
    num_dataset = DATASET
    data = Dataset(num_dataset) # Retrieve dataset object
    # Initialize IterReg feature
    algo = ALGO
    logger = IterReg(algo,num_dataset) # Init logger GDSVM

    # Print Header
    print("-- Using dataset {} | ALGO: {} --".format(num_dataset, algo))

    # Load weights
    try:
        w = logger.load() # Load saved weights
    except FileNotFoundError:
        w = np.zeros(data.X_tr.shape[1]) # Init to zeros
    loss_gd = get_loss_SVM(data.X_tr, data.y_tr, w) # Comp cost fun
    lamb = LAMB_ARR[num_dataset-1] # Regularization parameter
    print("Hot-start Loss Value: {}".format(loss_gd.round(2)))
    print("Using lambda: {}".format(lamb))
    print("Press Ctrl+C to stop and show results")
    print("Iterating...")
    tau = 1/(np.linalg.norm(data.X_tr, 2)**2) # Step size
    descent = 1 # Init
    tol = 1e-8 # Convergence tolerance
    while(abs(descent) > tol): # Converge when within tol
        try:
            w_new = step_GDSVM(data.X_tr, data.y_tr, w, lamb, tau)
            w = w_new
            # Check convergence
            loss_gd_new = get_loss_SVM(data.X_tr, data.y_tr, w)
            descent = loss_gd - loss_gd_new
            loss_gd = loss_gd_new
        except KeyboardInterrupt:
            break
    logger.save(w) # Save progress

    # Compute metrics with new w
    y_hat = classify(data.X,w) # Classify test set
    perr = get_perr(y_hat,data.y) # Get percent error
    loss_gd = get_loss_SVM(data.X_tr, data.y_tr, w) # Comp cost fun

    # Output results
    print("\nGradient Descent Support Vector Machine classification:")
    print("Percent labels misclassified: {}%".format(perr.round(2)))
    print("Training Loss Value: {}".format(loss_gd.round(2)))
