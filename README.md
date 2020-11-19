# JuliodDeepLearning

This project implements a deep learning library from scratch. It currenly only supports CPU computations and provides the following building blocks:

* nn::DenseLayer: Fully connected layer
* nn::CrossEntropy
* nn::MSE 
* nn::ReluLayer
* nn::SoftmaxLayer

# Build

We provide a conda environment definition file and scripts to build a Docker image that creates a conda environment out of it:

```
./build.sh
./dev.sh
cd /opt/development
emacs bin/app.cpp
```


