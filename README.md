## Mnist annihilator

# Overview

This is a extremely-lightweight Convolutional Neural Network that achieves >99% training accuracy (~98% test accuracy) on the mnist dataset.
This CNN has a extremely low amount of parameters (2114) and, thanks to parameter-count-reducing techniques, keeps a relatively high efficiency.

# Parameter-count-reducing techniques 

* Using Depthwise Convolutional layers instead of traditional convolutional layers
* Using traditional Convolutional layers with dilation rates to lower the kernel size (and thus, the amount of parameters) while keeping the same receptive field.
* Reducing the amount of filters right before the Flatten layer
* Removing Batch Normalization layers

# How to use 

Replace the training part in the <i>mnist_annihilator</i> function and load the model instead. Then feel free to test/benchmark/try to improve the model.
However, this initially was not made for direct reuse, but to showcase different ways to reduce the parameters amount of a Convolutional Neural Network classifier.
