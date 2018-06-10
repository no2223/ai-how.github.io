---
layout: post
title: Input pipeline for deep learning experiments in Keras, Tensorflow and Pytorch 
published: true
comments: true
---
# Quick Intro to Cyclic Learning Rate (CLR)

Without spending too much time in explaining Learning rate, I will straight forward jump to Cyclic Learning Rate (CLR) proposed in [1](https://arxiv.org/pdf/1506.01186.pdf). CLR changes the learning rate cyclically from low to high and back to low in a given bounds, this helps rapid traversal of saddle points (when learning rate increases) resulting in reaching higher accuracy faster and improved model performance. Shown below is the depiction of how learning rate changes cyclically where step size is the number of iterations considered in changing the learning rate from low to high. As the author states a cycle consists of approx 2-8 number of epochs.

<p align="center"> <img src="https://ai-how.github.io/img/CLR.png" width="250" height="200" /> </p>

# finding the min and max bounds

To identify the min and max learning rates, the paper suggest increasing the learning rate from low to some high value and monitor the loss, point where loss (or accuracy) starts to increase (or decrease for accuracy metrics) should be the bound for max learning rate. Below is an experiment I conducted to demonstrate this for SVHN dataset. Here the learning rate is changed linearly from .001 to .5 for 150 number of epochs.

<p align="center"> <img src="https://ai-how.github.io/img/min_max.png" width="600" height="600" /> </p>
