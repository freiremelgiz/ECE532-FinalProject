#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


"""
This class provides an interface to logging iteration results.
The results can also be loaded to provide a hot-start to
iterative methods like Gradient Descent to speed up convergence

Constructor:
algo - string identifier for algorithm used
dataset - integer representing which dataset to use [1,6]

algo chart:
GDLS - Gradient Descent for Least Squares problem
GDHL - Gradient Descent for Hinge Loss problem
GDSVM - Gradient Descent for Support Vector Machine
SGDNN - Stochastic Gradient Descent for Neural Network
"""

# List of algorithm codes available
ALGOS = ('GDLS','GDHL','GDSVM','SGDNN')

import numpy as np
from os.path import realpath # Absolute path

# Logging class to provide hot-start to iterative methods
class IterReg():
    def __init__(self,algo,dataset):
        # Check dataset and algo validity
        if dataset > 6 or dataset < 1:
            raise RuntimeError("Cannot find dataset {}".format(dataset))
        if algo not in ALGOS:
            raise RuntimeError("Code {} is not implemented".format(algo))

        # Build file path
        script_path = realpath(__file__)[:-10] # Get abs path
        self.fpath = script_path + '../resources/log/iter_'+ algo+'_'+str(dataset)+'.log'

        # Use correct encoding - Legacy compatibility
        if 'SGDNN'== algo:
            self.save = self._save_NN
            self.load = self._load_NN
        else:
            self.save = self._save_convex
            self.load = self._load_convex

    # Read log file into array of floats
    def _read_arr(self):
        flog = open(self.fpath) # Open file for reading
        raw = flog.read()
        flog.close() # Close file
        str_arr = raw.strip('[]').split('\n') # Strip and Split
        arr = np.array([float(i) for i in str_arr[:-1]]) # Convert to arr
        return(arr)

    # Save array into log file
    def _save_arr(self, data):
        np.savetxt(self.fpath,data)

    # Save data (numpy array) in logfile for convex loss functions
    def _save_convex(self, data):
        self._save_arr(data)

    # Load computed value in logfile for convex loss functions
    def _load_convex(self):
        return self._read_arr()

    # Save data in logfile for Neural Network
    def _save_NN(self, loss_new, nodes_new, W, v):
        try:
            arr = self._read_arr() # Read array from log file
            loss = arr[0] # Loss value is 1st element
            nodes = int(arr[1]) # Number of nodes is 2nd element

            # Check if strictly less loss
            if(loss_new >= loss):
                return 1 # Exit with code (No action)
        except FileNotFoundError: # Save if no file exists
            pass

        coded = W.reshape(W.shape[0]*W.shape[1]) # Flatten W
        coded = np.hstack((loss_new, nodes_new, coded, v[:,0])) # Encode
        self._save_arr(coded) # Save to log
        return 0

    # Load weights from logfile for Neural Network
    def _load_NN(self):
        arr = self._load_convex()
        nodes = int(arr[1]) # Number of nodes is 2nd element
        v = arr[-nodes:].reshape((nodes,1)) # Last nodes entries are v
        W = arr[2:-nodes] # Extract flattened W
        W = W.reshape((nodes,int(len(W)/nodes))) # Reshape into rxn
        return W, v

    # Load the best loss function of Neural Network SGD
    def load_loss(self):
        arr = self._read_arr() # Read array from log file
        loss = arr[0] # Loss value is 1st element
        return loss

    # Load the number of nodes used on log file
    def load_nodes(self):
        arr = self._read_arr() # Read array from log file
        nodes = int(arr[1]) # Number of nodes is 2nd element
        return nodes
