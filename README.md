# AI Optimization

This repository covers several techniques to optimize AI models. It is currently focusing only on reducing the parameters count of a model for hardware implementation.

## Overview

This is a extremely-lightweight Convolutional Neural Network that achieves >99% training accuracy (~98% test accuracy) on the mnist dataset (handwritten digits recognition).
This CNN has a extremely low amount of parameters (2114) and, thanks to parameter-count-reducing techniques, keeps a relatively high efficiency.

## Parameter-count-reducing techniques 

* Using Depthwise Convolutional layers instead of traditional convolutional layers
* Using traditional Convolutional layers with dilation rates to lower the kernel size (and thus, the amount of parameters) while keeping the same receptive field.
* Reducing the amount of filters right before the Flatten layer
* Removing Batch Normalization layers

## How to use 

Replace the training part in the <i>mnist_annihilator</i> function and load the model from the <i>annihilator.h5</i> file instead. Then feel free to test/benchmark/try to improve the model.

Keep in mind that this project was initially not made for direct reuse, but to showcase different ways to reduce the parameters amount of a Convolutional Neural Network-based classifier.
