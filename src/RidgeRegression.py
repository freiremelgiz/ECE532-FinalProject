#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


from Dataset import Dataset
from LeastSquares import get_perr
from LeastSquares import classify
import numpy as np
import matplotlib.pyplot as plt

""" Helper Functions """
# Returns the weights using the solution to Ridge Regression
def trainRidge(X, y, lamb):
    # Find least squares solution with training data provided
    w = np.linalg.inv(X.T@X + lamb*np.eye(X.shape[1]))@X.T@y
    return w


if __name__ == "__main__":
    # Initialize plot
    fig = plt.figure()
    for d in range(1,7):
        # Initialize a dataset
        num_dataset = d
        data = Dataset(num_dataset) # Retrieve dataset object
        print("-- Using dataset {} --".format(num_dataset))

        # Generate array of lambdas
        lamb_array = np.logspace(-6,1.3) # Create an array of lambda values

        # Initialize plotting arrays
        x_plot = np.zeros(len(lamb_array))
        y_plot = np.zeros(len(lamb_array))

        # Iterate over lambdas
        for i, lamb in enumerate(lamb_array):
            # Use helper functions to get percent error
            w = trainRidge(data.X_tr, data.y_tr, lamb) # Get weights with training set
            y_hat = classify(data.X,w) # Classify test set
            perr = get_perr(y_hat, data.y) # Get percent error
            # Store metric for plotting
            x_plot[i] = np.linalg.norm(w,2) # L2 norm of w
            y_plot[i] = perr # Percent error

        # Plot accumulated results
        ax = fig.add_subplot(2,3,d) # which subplot
        ax.plot(x_plot,y_plot)
        ax.set_title("Dataset {}".format(d), fontsize=10)
        ax.set_xlabel('$|\mathbf{w}|_2$',fontsize=8)
        ax.set_ylabel('Percent Error [%]',fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)

    # Show final figure
    plt.subplots_adjust(hspace=0.35, wspace=0.4)
    plt.show()


