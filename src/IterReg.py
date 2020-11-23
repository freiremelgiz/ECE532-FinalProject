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
"""

# List of algorithm codes available
ALGOS = ('GDLS','GDHL')

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

    # Save data (numpy array) in logfile
    def save(self, data):
        np.savetxt(self.fpath,data)

    # Load computed value in logfile
    def load(self):
        flog = open(self.fpath) # Open file for reading
        raw = flog.read()
        flog.close() # Close file
        str_arr = raw.strip('[]').split('\n') # Strip and Split
        arr = np.array([float(i) for i in str_arr[:-1]]) # Convert to arr
        return(arr)
