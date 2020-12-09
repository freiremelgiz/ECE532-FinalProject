#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"

"""
This script trains a neural network with one hidden layer
via Stochastic Gradient Descent.

Since the loss function being optimized is not
convex, the weights will only be stored if their
loss function is lower than the previously stored
weights.

"""

import numpy as np
from scipy.special import expit #Sigmoid activation function
from Dataset import Dataset
from IterReg import IterReg
from LeastSquares import get_perr
from LeastSquares import get_nerr

DATASET = 1
ALGO = 'SGDNN'


""" Helper Functions """
# Activation Function
def logsig(x):
    return expit(x)

# Take a step in SGD to train NN
def step_SGDNN(X,y,W,v,tau):
    i = np.random.randint(0,X.shape[0]) # Choose a random index
    x = X[i,:] # Extract selected feature
    y = y[i] # Extract selected label
    # Forward propagate
    H = logsig(W@x) # H is the array(h_1, h_2, ... h_r)
    y_hat = logsig(np.dot(H,v)) # Single output
    # Back propagate
    delta = (y_hat-y)*y_hat*(1-y_hat)
    v_new = v - np.array([tau*delta*H]).T # Weight update
    gamma = delta*v.T*H*(1-H) # gamma[rx1] r nodes
    W_new = W - tau*np.outer(gamma,x) # Weight update
    return W_new, v_new

# Classify datapoints using neural network
def classify_NN(X, W, v):
    H = logsig(W@X.T) # H[rxm] r nodes, m samples
    y_hat = logsig(H.T@v) # y_hat[mx1] m samples
    y_hat = np.sign(y_hat-0.5) # convert to [-1,+1] space
    return y_hat[:,0]

# Compute the loss function (Squared Error)
def get_loss_NN(X, y, W, v):
    H = logsig(W@X.T) # H[rxm] r nodes, m samples
    y_hat = logsig(H.T@v)[:,0] # y_hat[mx1] m samples
    loss = 0.5*np.sum((y_hat-y)**2)
    return loss

if __name__ == "__main__":
    # Initialize a dataset
    num_dataset = DATASET
    data = Dataset(num_dataset) # Retrieve dataset object
    # Initialize IterReg feature
    algo = ALGO
    logger = IterReg(algo,num_dataset) # Init logger GDSVM

    # Print Header
    print("-- Using dataset {} | ALGO: {} --".format(num_dataset, algo))

    # Expand feature vectors for bias term
    Xb = np.hstack((np.ones((data.X_tr.shape[0],1)), data.X_tr))
    nodes = 100 # Number of hidden nodes

    # Init weights (Non-convex, random init)
    W = np.random.randn(nodes,Xb.shape[1]) # Init to random
    v = np.random.randn(nodes,1) # Init to random

    # Load best loss function
    loss_nn_best = logger.load_loss()

    #print("Hot-start Loss Value: {}".format(loss_gd.round(2)))
    print("Press Ctrl+C to stop and show results")
    print("Iterating...")
    tau = 1/(np.linalg.norm(data.X_tr, 2)**2) # Step size
    tau = 0.001
    while(1): # Converge until user stop
        try:
            W_new, v_new = step_SGDNN(Xb, data.y_tr, W, v, tau)
            W = W_new
            v = v_new
            # Check for new minimum
            loss_nn_new = get_loss_NN(Xb, data.y_tr, W, v)
            if(loss_nn_new < loss_nn_best):
                print("New minimum found, continue iterating...")
                loss_nn_best = 0 #Display only once
        except KeyboardInterrupt:
            break

    print("\nStochastic Gradient Descent Neural Network classification:")
    discarded = logger.save(loss_nn_new, nodes, W, v) # Save progress

    # Check if progress was logged
    if discarded:
        print("Discarded loss: {}".format(loss_nn_new.round(2)))
        W, v = logger.load() # Load better weights

    # Compute metrics with new W
    # Expand feature vectors for bias term
    X = np.hstack((np.ones((data.X.shape[0],1)), data.X))
    y_hat = classify_NN(X,W,v) # Classify test set
    perr = get_perr(y_hat,data.y) # Get percent error
    loss_nn = get_loss_NN(Xb, data.y_tr, W, v) # Comp cost fun

    # Output results
    if not discarded:
        print("New local minimum found, weights logged")
    print("Best percent labels misclassified: {}%".format(perr.round(2)))
    print("Best training loss: {}".format(loss_nn.round(2)))
