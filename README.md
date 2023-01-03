# DAGMM-Needle
Project: Needle (NEcessary Elements of Deep Learning) Implementation for the Deep Autoencoding Gaussian Mixture Model (DAGMM)

Authors: Zefang Liu, Bin Gao, Chunshan Ma

## Introduction

In this project, we implement new features of matrix inverse and Cholesky decomposition by using the developed library in the [Deep Learning Systems](https://dlsyscourse.org/) course, Needle (NEcessary Elements of Deep Learning), plus the implementation of the Deep Autoencoding Gaussian Mixture Model ([DAGMM](https://openreview.net/forum?id=BJJLHbb0-)). The DAGMM is applied to build a network intrusion detector for the [KDD CUP 1999 Dataset](https://archive.ics.uci.edu/ml/datasets/kdd+cup+1999+data) with the capability of distinguishing between anomalous intrusion connections and benign regular connections.

## Running

To run the Needle-DAGMM, please go to the notebook [`project.ipynb`](https://github.com/zefang-liu/DAGMM-Needle/blob/main/project.ipynb). Please save a copy of the notebook in your drive before running it in the Colab. The notebook will clone this repository to your drive.

## Needle Framework

Needle is a lightweight PyTorch-like deep learning library. In the [course assignments](https://dlsyscourse.org/assignments/), we have implemented automatic differentiation, gradient descent optimizers, standard operators, and linear algebra backends on both CPU and GPU devices. In this project, we add several new linear algebra operators, including matrix inverse, matrix determinant, and Cholesky decomposition.
