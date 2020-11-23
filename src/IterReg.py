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

Instance variables:

"""

import numpy as np # TODO Remove dependency
from os.path import realpath # Absolute path

# Logging class to provide hot-start to iterative methods
class IterReg():
    def __init__(self,algo,dataset):
        # Check dataset validity
        if dataset > 6 or dataset < 1:
            raise RuntimeError("Cannot find dataset {}".format(dataset))
        # TODO Check algo list
        script_path = realpath(__file__)[:-10] # Get abs path
        # Build file path
        self.fpath = script_path + '../resources/log/iter_'+ algo+str(dataset)+'.log'

        # Store algo for save-dependent features
        self.algo = algo


    # Save data (numpy array) in logfile
    def save(self, data):
        #flog = open(self.fpath,'w') # Open file for writing
        #flog.write(str(data)) # Write data
        np.savetxt(self.fpath,data)
        #flog.close() # Close file

    # Load computed value in logfile
    def load(self):
        flog = open(self.fpath) # Open file for reading
        raw = flog.read()
        flog.close() # Close file
        str_arr = raw.strip('[]').split('\n') # Strip and Split
        arr = np.array([float(i) for i in str_arr[:-1]])
        return(arr)

# Test class
if __name__ == "__main__":
    it = IterReg('GD',1)
    #print(it.load())
    a = np.ones(5)
    it.save(a)



