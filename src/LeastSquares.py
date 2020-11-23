#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


from Dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from IterReg import IterReg # Iterative register worker

""" Helper Functions """
# Perform data classification with provided weights
def classify(X,w):
    y_hat = np.sign(X@w) # Predict labels
    return y_hat

# Gets the percent error of the predictions
def get_perr(y_hat, y):
    nerr = get_nerr(y_hat,y) # Use other function
    perr = nerr/len(y)*100 # Percent error
    return perr

# Gets the number of misclassifications
def get_nerr(y_hat, y):
    nerr = np.sum(np.abs(y_hat-y)/2) # Number of errors
    return nerr

# Returns the weights using the original LS solution
def trainLSOrig(X, y):
    # Find least squares solution with training data provided
    w = np.linalg.inv(X.T@X)@X.T@y
    return w

# Compute the loss function value Mean Squared Error
def get_loss_MSE(X, y, w): # Compute cost func
    loss = np.linalg.norm(X@w-y,2)**2 # MSE
    return loss

# Returns the weight using Gradient Descent on LS cost function
def trainLSGrad(X, y):
    w_cf = np.linalg.inv(X.T@X)@X.T@y # Closed form solution to LS
    loss_cf = np.linalg.norm(X@w_cf-y,2)**2 # Loss function with LS solution
    tol = 1 # Tolerance for convergence
    tau = 1/(np.linalg.norm(X,2)**2) # Step size
    print("Step size: {}".format(tau))
    w = np.zeros(X.shape[1]) # Init to zeros
#    w = w + 20
    loss = np.linalg.norm(X@w-y,2)**2 # Initial loss
    v = np.zeros(X.shape[1])
    while loss > (loss_cf + tol): # Iterate 20 times
        grad = 2*X.T@(X@w - y) # Compute gradient
        w_new = w - tau*grad # Gradient Descent
        loss = np.linalg.norm(X@w_new-y,2)**2
        print(loss)
        #print(np.linalg.norm(w_new,2))
#        v_new = gamma*v + tau*grad
#        w_new = w - v_new
        w = w_new
#        v = v_new
    return w

# Take a step in GD
def step(X, y, w, tau):
    grad = 2*X.T@(X@w - y) # Compute gradient
    w_new = w - tau*grad
    return w_new

# Stochastic Gradient Descent
def trainLS_SGD(X, y):
    w_cf = np.linalg.inv(X.T@X)@X.T@y # Closed form solution to LS
    loss_cf = np.linalg.norm(X@w_cf-y,2)**2 # Loss function with LS solution
    print(loss_cf)
    tol = 1 # Tolerance for convergence
    tau = 1/(np.linalg.norm(X,2)**2) # Step size
    print("Step size: {}".format(tau))
    w = np.zeros(X.shape[1]) # Init to zeros
    loss = np.linalg.norm(X@w-y,2)**2 # Initial loss
    # MOMENTUM
    v = np.zeros(X.shape[1])
    gamma = 0.9
    ##
    while loss > (loss_cf + tol): # Iterate 20 times
        i = np.random.randint(0,X.shape[0]) # Generate random index
        grad = 2*X[i,:]*(np.dot(X[i,:],w) - y[i]) # Compute sg
        #w_new = w - tau*grad # Gradient Descent
        #print(np.linalg.norm(w_new,2))
        v_new = gamma*v + tau*grad
        w_new = w - v_new
        # Convergence metric
        loss = np.linalg.norm(X@w_new-y,2)**2
        print(loss)
        # Reg Shift
        w = w_new
        v = v_new
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
    loss_tr = get_loss_MSE(data.X_tr, data.y_tr, w) # Compute cost func

    # Output results
    print("Original Least Squares classification:")
    print("Percent labels misclassified: {}%".format(perr.round(2)))
    print("Training Loss Value: {}".format(loss_tr.round(2)))


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
    loss_r = get_loss_MSE(X_r, data.y_tr, w) # Compute cost func

    # Output results
    print("Least Squares with Rank-{} approximation:".format(rank))
    print("Percent labels misclassified: {}%".format(perr.round(2)))
    print("Training Loss Value: {}".format(loss_r.round(2)))



    # Compare Gradient Descent with Closed Form
    print("\n-- Iterating Gradient Descent --")
    logger = IterReg('GD',num_dataset) # Init logger
    # Load weights
    try:
        w = logger.load() # Load saved weights
    except FileNotFoundError:
        w = np.zeros(data.X_tr.shape[1]) # Init to zeros
    loss_gd = get_loss_MSE(data.X_tr, data.y_tr, w) # Comp cost fun
    print("Current Loss Value: {}".format(loss_gd))
    print("Press Ctrl+C to stop and show results")
    tau = 1/(np.linalg.norm(data.X_tr,2)**2) # Step size
    while((loss_gd-loss_tr) > 1): # While not converged
        try:
            w_new = step(data.X_tr, data.y_tr, w, tau)
            w = w_new
        except KeyboardInterrupt:
            break
    logger.save(w) # Save progress

    # Compute metrics with new w
    y_hat = classify(data.X,w) # Classify test set
    perr = get_perr(y_hat, data.y) # Get percent error
    loss_gd = get_loss_MSE(data.X_tr, data.y_tr, w) # Comp cost fun

    # Output results
    print("\nGradient Descent Least Squares classification:")
    print("Percent labels misclassified: {}%".format(perr.round(2)))
    print("Training Loss Value: {}".format(loss_gd.round(2)))
