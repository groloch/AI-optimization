## Mnist annihilator

# Overview

This is a very-lightweight Convolutional Neural Network that achieves >99% training accuracy (~98% test accuracy) on the mnist dataset.
This CNN has a extremely low amount of parameters (2114) and, thanks to parameter-count-reducing techniques, keeps a relatively high efficiency.
Parameter-count-reducing techniques used:
* Using Depthwise convolutional layers instead of traditional convolutional layers
* Using Convolutional layers with dilation rates to lower the kernel size (and thus, the amount of parameters) while keeping the same receptive field.
* Reducing the amount of filters right before the Flatten layer
* Removing Batch Normalization layers
