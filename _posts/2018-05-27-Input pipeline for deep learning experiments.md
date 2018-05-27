---
layout: post
title: Input pipeline for deep learning experiments: Keras, Tensorflow and Pytorch 
published: false
comments: true
---

Increasing list of algorithms and techniques aiming to improve the performance of deep learning models often instills a curiosity to benchmark how well these models perform. Benchmarking these techniques (on a dataset specific to business) often require writing your own pipeline which could quickly fetch mini-batches and run multiple iterations to search for optimal hyper parameters.

A quick and dirty practice is to load your training data into RAM using numpy and pandas functionalities (np.laod or pd.read_csv). This works well only if the dataset is small enough to fit in the memory. From a personal experience this slows down the entire training process resulting in longer model development and evaluation cycle. This blog post describes how you can quickly write input pipeline on a platform of your own choice:

