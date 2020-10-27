#! /usr/bin/env python3

__author__ = "Victor Freire"
__email__ = "freiremelgiz@wisc.edu"


from Dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt

# Initialize a dataset
num_dataset = 1
data = Dataset(num_dataset) # Test dataset 1

# Remove mean from data
#print(data.X.shape)
#print(np.mean(data.X,0).shape)
X_m = data.X - np.mean(data.X,0)

# Find skinny SVD of data matrix
U, s, VT = np.linalg.svd(X_m,full_matrices=False)

# Plot singular values vs features
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.log10(s))
ax.set_xlabel('Sing value index $i$', fontsize=16)
ax.set_ylabel('$\log_{10}(\sigma_i)$', fontsize=16)
ax.set_title('Dataset {} Singular Values'.format(num_dataset), fontsize=18)
plt.show()

