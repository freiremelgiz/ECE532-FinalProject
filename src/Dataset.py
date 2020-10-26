#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


"""
This class holds the chosen dataset and allows for
easy manipulation
"""


from scipy.io import loadmat

# Dataset class holding one dataset
class Dataset():
    def __init__(self):
        data = loadmat('./../resources/data/pub_dataset1.mat')

