*version:* 1.0.0

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

<img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/dataset_dim.PNG">

<img src="https://github.com/freiremelgiz/ECE532_FinalProject/blob/master/resources/img/feature_table.PNG">
Figure 1. Feature description based on raw data. The features are found in the dataset along with labels.

## Authors

* [**Victor Freire**](mailto:freiremelgiz@wisc.edu) - [University of Wisconsin-Madison](https://www.wisc.edu/)

## References

[Liang Zhao](mailto:lzhao9@gmu.edu) - George Mason University - [Dataset](http://mason.gmu.edu/~lzhao9/materials/data/UAV/)

A. Alipour-Fanid, M. Dabaghchian, N. Wang, P. Wang, L. Zhao and K. Zeng, 'Machine Learning-Based Delay-Aware UAV Detection and Operation Mode Identification Over Encrypted Wi-Fi Traffic,' in IEEE Transactions on Information Forensics and Security, vol. 15, pp. 2346-2360, 2020.
