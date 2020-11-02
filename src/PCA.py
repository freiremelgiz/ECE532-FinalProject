#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


from Dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt


fig_log = plt.figure()
ax_log = fig_log.add_subplot(111)
fig_lin = plt.figure()
ax_lin = fig_lin.add_subplot(111)

# Initialize the datasets
for i in range(1,7): # Six datasets
    num_dataset = i
    data = Dataset(num_dataset) # Test dataset 1

    # Remove mean from data
    #print(data.X.shape)
    #print(np.mean(data.X,0).shape)
    X_m = data.X - np.mean(data.X,0)

    # Find skinny SVD of data matrix
    U, s, VT = np.linalg.svd(X_m,full_matrices=False)

    # Since data is in the rows, PCA is v_1
    print("PC of dataset {}: \n{}".format(i,VT.T[:,0].round(2)))

    # Plot singular values vs features
    ax_log.plot(np.log10(s))
    ax_lin.plot(s)



ax_lin.set_xlabel('Sing value index $i$', fontsize=16)
ax_lin.set_ylabel('$\sigma_i$', fontsize=16)
ax_lin.set_title('Singular Values', fontsize=18)
ax_lin.legend(['Dataset 1','Dataset 2','Dataset 3','Dataset 4', 'Dataset 5', 'Dataset 6'])

ax_log.set_xlabel('Sing value index $i$', fontsize=16)
ax_log.set_ylabel('$\log_{10}(\sigma_i)$', fontsize=16)
ax_log.set_title('Singular Values', fontsize=18)
ax_log.legend(['Dataset 1','Dataset 2','Dataset 3','Dataset 4', 'Dataset 5', 'Dataset 6'])

plt.show()



