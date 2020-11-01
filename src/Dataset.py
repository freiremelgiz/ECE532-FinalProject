#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


"""
This class holds the chosen dataset and allows for
easy manipulation. There are 6 datasets.

Constructor:
dataset - integer representing which dataset to use [1,6]

Instance variabls:
self.X - Data matrix, each row is a datapoint
self.y - Data labels, +1 for a quadcopter, 0 otherwise
self.rows - Number of datapoints
self.cols - Number of features
Each dataset will have a subset (_tr) of training data
self.X_tr - Training data matrix
self.y_tr - Training data labels
self.rows_tr - Number of training datapoints
"""


from scipy.io import loadmat

# Dataset class holding one dataset
class Dataset():
    def __init__(self,dataset):
        data = loadmat('./../resources/data/pub_dataset' + str(dataset) + '.mat')
        self._process_dataset(data) # Obtain the data matrices and labels
        self._get_dimensions()      # Find dimensions of the data
        self._convert_labels()      # Convert the labels to +1 -1 space

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

