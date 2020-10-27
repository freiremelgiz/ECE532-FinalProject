*version:* 1.1.0

# ECE532 Final Project - Classifying a public dataset

This repository contains the code for the final project of ECE532: Matrix Methods in Machine Learning. The final project consists on choosing a public dataset and classifying it using algorithms learned in the class.

## Project Dataset - Unmmanned Aerial Vehicles (UAV) Intrusion Detection
The dataset I chose was posted at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Unmanned+Aerial+Vehicle+%28UAV%29+Intrusion+Detection).
However, the dataset can be downloaded from the author's personal [website](https://archive.ics.uci.edu/ml/datasets/Unmanned+Aerial+Vehicle+%28UAV%29+Intrusion+Detection).
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

The Singular Value Decomposition of dataset 1 yields the singular values plotted in the images below.
<img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/PCA_sigma.png" width="400"> | <img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/PCA_sigma_log.png" width="400">

![formula](https://render.githubusercontent.com/render/math?math=\mathbf{x}=ab^i)

## Project Timeline
* **10/22/2020** Project Proposal Due
* 10/26/2020  Principal Component Analysis
* 11/02/2020  Least Squares Classification
* 11/09/2020  SVD and Tikhonov Regularization
* **11/17/2020** Update 1 Due
* 11/23/2020  Hinge Loss Classification
* **12/01/2020** Update 2 Due
* 12/05/2020  Neural Network Classification
* **12/12/2020** Final Report Due
* **12/17/2020** Peer Review Due

## Authors

* [**Victor Freire**](mailto:freiremelgiz@wisc.edu) - [University of Wisconsin-Madison](https://www.wisc.edu/)

## References

[Liang Zhao](mailto:lzhao9@gmu.edu) - George Mason University - [Dataset](http://mason.gmu.edu/~lzhao9/materials/data/UAV/)

A. Alipour-Fanid, M. Dabaghchian, N. Wang, P. Wang, L. Zhao and K. Zeng, 'Machine Learning-Based Delay-Aware UAV Detection and Operation Mode Identification Over Encrypted Wi-Fi Traffic,' in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 2346-2360, 2020.
