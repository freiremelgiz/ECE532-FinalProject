# ECE532 Final Project - Classifying a public dataset

This repository contains the code for the final project of ECE532: Matrix Methods in Machine Learning. The final project consists on choosing a public dataset and classifying it using algorithms learned in the class.

## Project Dataset - Unmmanned Aerial Vehicles (UAV) Intrusion Detection
The dataset I chose was posted at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Unmanned+Aerial+Vehicle+%28UAV%29+Intrusion+Detection).
However, the dataset can be downloaded from the author's personal [website](http://mason.gmu.edu/~lzhao9/materials/data/UAV/).
The dataset was collected for 3 quadcopter models and 2 data flow directions for each (6 datasets).
Each dataset also contains two matrices of "testing data" and "training data".
Each matrix consists of 9 statistic measurements of network traffic timing and size for a total of 18 features. The data collection was performed in both unidirectional and bidirectional data flow.
The bidirectional flow datasets include 3x more features because the data comes from uplink, downlink and a combined total.
Figure 1 shows a mathematical description of each feature.
The datasets are also labeled with the correct device type. Traffic features coming from a UAV are labeled with +1. Other features are labeled as 0. Table 1 summarizes the dimensions of each dataset.

<p align="center">
  <img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/feature_table.PNG" alt="Feature table" width="600">
</p>

**Figure 1**. Feature description based on raw data. The features are found in the dataset along with labels.

**Table 1**. Dataset dimension summary.
<p align="center">
  <img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/dataset_dim.PNG" alt="dataset dimensions" width="400">
</p>

## Algorithms that will be implemented
Since the features include some niche statistical calculations, I will preform principal component
analysis on the testing data to understand which statistical measures are more significant. This may
help me decide to perform low-rank approximation if certain features are meaningless to avoid small
singular values in my SVD.
* I will employ a basic binary linear classifier of the form y=sign(Xw). Where +1 corresponds to a UAV device type and −1 corresponds to non-UAV type. I will use a least-squares problem to find the weights using the training datasets. I will then validate the classifier on the testing data provided. However, since the testing and training data were separated a priori, I may attempt cross-validation on the provided training set to prevent overfitting. Since dataset6 has more features than datapoints in the training set, I will use a low-rank approximation using SVD along with Tikhonov Regularization to train the classifier while minimizing noise amplification. A parameter whose performance I will investigate in this problem is the λ for the regularization term to place importance on minimizing the norm of the weights.

* The second algorithm I will use to classify the dataset is Hinge Loss. We will learn about this algorithm in Week 11 of the class. This algorithm involves a loss function used for training the classifier.

* The third algorithm I will use to classify the dataset is neural networks. We will learn about this algorithm in Week 13 of the class. Neural networks pass the input features through certain nodes with weights. I predict a key parameter in this algorithm will be the number of"nodes" in the network.

## PCA - Principal Component Analysis
When trying to import the data from the `.mat` files into Python3, I found that the dataset was saved with MATLAB v7.3.
The `scipy.io.loadmat()` function cannot handle `.mat` files that were saved with this version of MATLAB.
To solve this problem, I opened each dataset in MATLAB R2020a (which can handle the old files) and re-saved them in a newer format.
The updated, but unmodified dataset files can be found in `~/resources/data/`.

<img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/PCA_sigma.png" height="250"> <img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/PCA_sigma_log.png" height="250">

**Figure 2**. Singular values of testing data plotted against their index (left) and on a log scale (right).

The magnitude of the singular values decreases rapidly. However, values remain relatively high until about index 20 for the first 3 datasets, and index 8 for the last 3 datasets. I will consider this and perform a low-rank approximation of the data matrix as  

![formula](https://render.githubusercontent.com/render/math?math=\mathbf{X}_r=\sum_{i=1}^r\mathbf{u}_i\sigma_i\mathbf{v}_i^T)

## Least Squares Classification
I added a processing step to the Dataset class to convert the label vectors from the {0,+1} space to the {-1,+1} space to match the output of the sign() function.

The first classification was a simple Least Squares training the weight vector with the training data provided in the datasets. Then I classified the testing data on each dataset. Using the results from Principal Component Analysis, I decided to perform the Least Squares classification on a low-rank approximation of the training data matrix. The rank was chosen based on the trend of singular values found. The results are summarized in the table below.

**Table 2**. Least Squares classification results. Also shows results when the weights were trained on a low-rank approximation of the training data matrix. The rank of the approximation was informed by the PCA.

| Dataset |  Error: Full-rank  | Approximation rank |   Error: Low-rank  |
| :----:  |  :--------------:  | :----------------: |  :---------------: |
|   1     |         0.20 %     |  	 20	    |       32.08 %      |
|   2     |         0.11 %     |         20         |       36.90 %      |
|   3     |         1.78 %     |         20         |       29.62 %  	 |
|   4     |         3.25 %     |          8         |       54.58 %   	 |
|   5     |         8.61 %     |          8         |       44.61 %      |
|   6     |        56.06 %     |          5         |       81.82 %      |

Aside from dataset 6, which has very scarce data, the percent errors are very low for the original Least Squares problem with the full-rank training data. The low-rank approximation did not perform well. This suggests that a large number of features are important in the classification process.

## Tikhonov Regularization (Ridge Regression)
Sometimes, the unregulated Least Squares problem can lead to classifier weights with large norms. This tends to cause unwanted noise amplification when performing classification of future data. Regularization techniques modify the Least Squares minimization problem by adding a regularizing term. In the case of Ridge Regression, the regularizing term is the ![formula](https://render.githubusercontent.com/render/math?math=\lambda)-parameterized L2 norm of the classifier weights.  


![formula](https://render.githubusercontent.com/render/math?math=min_\mathbf{w}||\mathbf{Xw}-\mathbf{y}||_2^2%2B\lambda||\mathbf{w}||_2^2)  


I solved the Ridge Regression problem for each dataset using the provided training data. The ![formula](https://render.githubusercontent.com/render/math?math=\lambda) parameter was varied between \[1e-6, 20\].
For each value, a different classifier weight vector was computed and used to classify the testing data.
The Figure 3 shows the percent error evolution as the L2 norm of the weight vector increases.
In general, the percent error is expected to increase for low weight vector norms because the minimization burden is shifted from the error norm.
However, sometimes a large weight vector norm amplifies noise in the feature measurements and results in larger percent error.

<img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/Ridge.png" height="500">

**Figure 3**. Classification percent error on test data plotted against the L2 norm of the weight vector.

In order to select the best ![formula](https://render.githubusercontent.com/render/math?math=\lambda) parameter for each dataset, I held out 10% of the testing data to perform cross-validation across all the found classifiers. The ![formula](https://render.githubusercontent.com/render/math?math=\lambda) that performs best (least misclassifications) is selected and used on the rest of the testing data. The final performance of Ridge Regression for each dataset is summarized in the table below.

**Table 3**. Tikhonov Regularization classification results. Along with cross-validation performance parameters.

| Dataset |  Error  |  ![formula](https://render.githubusercontent.com/render/math?math=\lambda) | Cross-Validation set size | Error: Holdout |
| :----:  |  :---:  | :-----: | :---: | :----: |
|   1     |  0.28 % |    1e-6 |  1763 | 1.13 % |
|   2     |  0.04 % |   19.95 |  1569 | 0.00 % |
|   3     |  2.62 % |   19.95 |   500 | 0.00 % |
|   4     |  3.52 % |  1.6e-5 |  1060 | 0.75 % |
|   5     |  9.97 % |  1.4e-6 |  1351 | 0.74 % |
|   6     | 50.85 % |   19.95 |     7 | 0.00 % |

Overall, the classifier performance decreased when using the regularized least squares classifier. Except for dataset 6 which showed a slight classification improvement when the L-2 norm of the classifier was reduced.

## Iterative Methods (Gradient Descent)
When I first started implementing Gradient Descent algorithms I found out that convergence would be very slow for the selected datasets. For this reason, I developed a logging system to save convergence progress via gradient descent.
The logging system is implemented in `~/src/IterReg.py`. This class provides an interface to save the most recently computed weights in a `.log` file. The class also provides a function to read these weights to serve as a hot-start for the gradient descent algorithm in future script runs. This way, I can save gradient descent progress and improve convergence.
The logged weight vectors for each algorithm and dataset are all stored in `~/resources/log/`. And follow the naming convention `iter_ALGO_DATASET.log`. Where "ALGO" is the algorithm code and "DATASET" is the dataset number for these weights.

<p align="center">
  <img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/example_IterReg.PNG" alt="IterReg" width="600">
</p>

**Figure 4**. Example run of `HingeLoss.py` for dataset 1 showing how the loss function value decreases after some time iterating with the Gradient descent algorithm. When the program is run again, it will start the iteration with the last computed weight vector as the starting point.

**Table 4**. Algorithm `ALGO` code description. These descriptions are also provided in `IterReg.py`. The program also checks to make sure the algorithm code is valid and implemented.

| ALGO |   Description  |
| :----:  |  :----:  |
|  GDLS   |  Gradient Descent for Least Squares problem  |
|  GDHL   |  Gradient Descent for Hinge Loss problem  |
|  GDSVM  |  Gradient Descent for Support Vector Machine problem  |
|  SGDNN  |  Stochastic Gradient Descent for Neural Network training  |


## Hinge Loss and Support Vector Machine
The Hinge Loss cost function addresses the limitation of the Mean Squared Error cost function with regard to data outliers.
In other words, Hinge Loss is insensitive to datapoints which are far away from the decision boundary.

![formula](https://render.githubusercontent.com/render/math?math=min_\mathbf{w}\sum_{i=1}^N%281-y_i\mathbf{x}_i\mathbf{w}%29_%2B)

The Hinge Loss cost function is convex. It has one non-differentiable point. This point is addressed with the concept of sub-gradients. I chose the sub-gradient to be 0 at this point because this is a popular choice and leads to a compact gradient definition. I solved the minimization problem with Gradient Descent, leveraging the `IterReg.py` architecture to save convergence progress.


The Support Vector Machine classification algorithm uses the Hinge Loss cost function with a L2 norm regularization term. This regularization term ensures that the minimization converges to the max-margin decision boundary. This is the boundary at the maximum distance from the closest datapoints.

![formula](https://render.githubusercontent.com/render/math?math=min_\mathbf{w}\sum_{i=1}^N%281-y_i\mathbf{x}_i\mathbf{w}%29_%2B%2B||\mathbf{w}||_2^2)

The Support Vector Machine problem was solved similarly to the Hinge Loss one via Gradient Descent. The regularization parameter ![formula](https://render.githubusercontent.com/render/math?math=\lambda) was chosen equal to the results of the Cross-Validation performed with Ridge Regression.


**Table 5**. Hinge Loss and Support Vector Machine classification results.

| Dataset |   Error: HL  |  Error: SVM  |  ![formula](https://render.githubusercontent.com/render/math?math=\lambda)  |
| :----:  |  :----:  |  :----:  | :-----: |
|   1     |  2.67 %  |  2.93 %  |       1e-6 |
|   2     |  0.00 %  |  0.00 %  |      19.95 |
|   3     |  2.34 %  |  2.22 %  |      19.95 |
|   4     |  7.76 %  |  6.70 %  |     1.6e-5 |
|   5     | 13.93 %  |  13.93 %  |     1.4e-6 |
|   6     | 36.36 %  | 36.36 %  |      19.95 |


## Non-Convex Optimization (IterReg)
Training a neural network with backpropagation using Stochastic Gradient Descent involves solving a non-convex optimization problem.
My previous implementation of the `IterReg` feature worked well for convex optimization where there is one global minimum.
However, in non-convex optimization there can be any number of local minimum where the solution may get stuck using gradient descent.
In practice, this problem is solved starting the training from random weights each time. This is a different demand than my prevous hot-start implementation of `IterReg`.

I modified the `IterReg` feature to check for better performing weights with respect to the cost function if this is a non-convex algorithm before storing the new set of weights into the log files.
In summary, the changes allow the training of neural networks to start from random values and converge to different minima each runtime. However, the `IterReg` feature will only store the best-performing weights.

## Neural Network Classification
A neural network with one hidden layer and enough nodes can be used to approximate any function. I use this theorem to train a neural network to approximate a complex decision boundary for my datasets.
The objective function I minimized to train the neural network is the Squared Error Loss:

![formula](https://render.githubusercontent.com/render/math?math=min_\mathbf{w}\sum_{i=1}^N\frac{1}{2}%28\hat{y}_i-y_i%29^2)


| Dataset |   Error: NN  |  Hidden Nodes  |
| :----:  |  :----:  |  :----:  |
|   1     |  2.67 %  |  200  |
|   2     |  0.00 %  |  200  |
|   3     |  2.34 %  |  200  |
|   4     |  7.76 %  |  200  |
|   5     | 13.93 %  |  200  |
|   6     | 16.67 %  |  200  |



## Project Timeline
* **10/22/2020** Project Proposal Due
* 10/26/2020  Principal Component Analysis
* 11/02/2020  Least Squares Classification
* 11/09/2020  Tikhonov Regularization
* **11/17/2020** Update 1 Due
* 11/23/2020  Hinge Loss Classification
* **12/01/2020** Update 2 Due
* 12/05/2020  Neural Network Classification
* **12/12/2020** Final Report Due
* **12/17/2020** Peer Review Due

## Authors

* [**Victor Freire**](mailto:freiremelgiz@wisc.edu) - [University of Wisconsin-Madison](https://www.wisc.edu/)

## References

[Liang Zhao](mailto:lzhao9@gmu.edu) - [George Mason University](https://www2.gmu.edu/) - [Dataset](http://mason.gmu.edu/~lzhao9/materials/data/UAV/)

A. Alipour-Fanid, M. Dabaghchian, N. Wang, P. Wang, L. Zhao and K. Zeng, 'Machine Learning-Based Delay-Aware UAV Detection and Operation Mode Identification Over Encrypted Wi-Fi Traffic,' in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 2346-2360, 2020.
