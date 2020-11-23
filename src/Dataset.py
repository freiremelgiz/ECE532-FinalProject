#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


"""
This class holds the chosen dataset and allows for
easy manipulation. There are 6 datasets.
Each dataset will have a subset (_tr) of training data
The training data matrix is also min-max normalized for GD

Constructor:
dataset - integer representing which dataset to use [1,6]

Instance variables:
self.X - Data matrix, each row is a datapoint
self.y - Data labels, +1 for a quadcopter, -1 otherwise
self.rows - Number of datapoints
self.cols - Number of features
self.X_tr - Training data matrix
self.X_trn - Normalized training data matrix
self.y_tr - Training data labels
self.rows_tr - Number of training datapoints
"""

import numpy as np # TODO Remove dependency
from scipy.io import loadmat # Read in dataset
from os.path import realpath # Absolute path

# Dataset class holding one dataset
class Dataset():
    def __init__(self,dataset):
        # Check dataset validity
        if dataset > 6 or dataset < 1:
            raise RuntimeError("Cannot find dataset {}".format(dataset))
        script_path = realpath(__file__)[:-10] # Get abs path
        # Load data file
        data = loadmat(script_path + '../resources/data/pub_dataset' + str(dataset) + '.mat')
        self._process_dataset(data) # Obtain the data matrices and labels
        self._get_dimensions()      # Find dimensions of the data
        self._convert_labels()      # Convert the labels to +1 -1 space
        self._normalize()           # Normalize training data for GD

    # Extract X matrices and y vectors from dataset
    def _process_dataset(self,datafile):
        data_te = datafile['data_te'] # Retrieve testing matrix
        data_tr = datafile['data_tr'] # Retrieve training matrix
        self.y = data_te[:,-1]        # Extract labels
        self.X = data_te[:,0:-1]      # Extract data matrix
        self.y_tr = data_tr[:,-1]     # Extract labels
        self.X_tr = data_tr[:,0:-1]   # Extract data matrix

    # Find the dimensions in the data matrices
    def _get_dimensions(self):
        self.rows = self.X.shape[0]
        self.cols = self.X.shape[1]
        self.rows_tr = self.X_tr.shape[0]

    # Convert labels from (0,1) to (-1,+1)
    def _convert_labels(self):
        self.y = 2*(self.y - 0.5)
        self.y_tr = 2*(self.y_tr - 0.5)

    # Normalize data for gradient descent (Symmetric descent)
    def _normalize(self):
        self.X_trn = (self.X_tr - np.mean(self.X_tr))/np.std(self.X_tr)
#        self.X_trn = np.zeros(self.X_tr.shape) # Init normalized mat
#        for j in range(self.cols):
#            rang = max(self.X_tr[:,j]) - min(self.X_tr[:,j]) # Range
#            shifted = self.X_tr[:,j]-min(self.X_tr[:,j])
#            self.X_trn[:,j] = shifted/rang


# Dataset inspection
if __name__ == "__main__":
    data = Dataset(4)
    print(np.mean(data.X_tr,0))
    # Compare norms of normalized and original training data
    print("Op Norm of training: {}".format(np.linalg.norm(data.X_tr,2)))
    print("Op Norm of normalized training: {}".format(np.linalg.norm(data.X_trn,2)))

