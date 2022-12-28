# Needle-DAGMM
Needle (NEcessary Elements of Deep Learning) Implementation for the Deep Autoencoding Gaussian Mixture Model (DAGMM)

Authors: Zefang Liu, Bin Gao, Chunshan Ma

## Introduction

In this project, we implement new features of matrix inverse and Cholesky decomposition by using the developed library in the [Deep Learning Systems](https://dlsyscourse.org/) course, Needle (NEcessary Elements of Deep Learning), plus an implementation of the Deep Autoencoding Gaussian Mixture Model ([DAGMM](https://openreview.net/forum?id=BJJLHbb0-)) model, taking advantage of these features. The Cholesky decomposition is a matrix decomposition, where a positive-definite Hermitian matrix is decomposed into a product of a lower triangular matrix with its conjugate transpose. The DAGMM is a model for unsupervised anomaly detection on multi- or high-dimensional data. The inverse of Cholesky decomposition is used in DAGMM to compute the inverse of the covariance matrix and the sample energy in the loss function. In this project, we apply the DAGMM to build a network intrusion detector for the [KDD CUP 1999 Dataset](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data) with the capability of distinguishing between anomalous intrusion connections and benign regular connections.

## Run the Needle-DAGMM

To run the code, please go to the notebook `project.ipynb`.
