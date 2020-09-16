#pragma once

#include <cstdarg>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "matrix.hpp"

namespace nn {

/*! \brief Base class for all layers of a neural network */
class Layer {
 public:
  virtual ~Layer(){};

  virtual void forward(m::Tensor& X) = 0;

  // grad_in: Tensor. Gradient coming from the "right"
  // in: Tensor. The input to the layer during the forward pass
  // grad_out: Tensor. The gradient after passing through this layer
  virtual void backward(m::Tensor& grad_in, m::Tensor& in, m::Tensor& grad_out) = 0;

  // This is only implemented by loss function "layers"
  virtual float loss(m::Tensor& X, m::Tensor& Y) { return 0.0; }

  virtual m::Tensor& output() = 0;
};

class CrossEntropy : public Layer {
 public:
  virtual ~CrossEntropy() {};

  // Not implemented for loss functions
  void forward(m::Tensor& X) override {}

  // grad_in: Tensor. Labels
  // in: Tensor. the output of the network
  // grad_out: Tensor. The gradient resulting from this layer
  void backward(m::Tensor& grad_in, m::Tensor& in, m::Tensor& grad_out) override {
              std::transform(in.data(), in.data() + in.rows() * in.cols(),
                             grad_in.data(),
                             grad_out.data(),
                             [](float a, float b) { return a - b; });
  }

  m::Tensor& output() override {};

  // Assumes Y is one hos encoded
  float loss(m::Tensor& X, m::Tensor& Y) override {
    float loss = 0.0;
    for (auto offset = 0 ; offset < X.rows() * X.cols() ; offset += X.cols()) {
      // Find the index of the class from the ground truth
      const auto begin = Y.data() + offset;
      const auto idx = std::find(begin, begin + Y.cols(), 1.0) - begin;

      // Accumulate the loss for the batch
      loss -= log2f(X.data()[offset + idx]);
    }
    return loss;
  }

  static std::shared_ptr<Layer> newLayer() {
    return std::make_shared<CrossEntropy>();
  }
};

class ReluLayer : public Layer {
 public:
  ReluLayer(std::size_t batch_size,
	    std::size_t input_dim) : a_{batch_size, input_dim} {}
  
  virtual ~ReluLayer() {};

  // Not implemented for loss functions
  void forward(m::Tensor& X) override {
    std::transform(X.data(), X.data() + X.rows() * X.cols(), a_.data(),
                   [](float a) { return std::max(0.0f, a); });
    std::cout << a_ << std::endl;
  }

  // grad_in: Tensor. Labels
  // in: Tensor. the output of the network
  // grad_out: Tensor. The gradient resulting from this layer
  void backward(m::Tensor& grad_in, m::Tensor& in, m::Tensor& grad_out) override {}

  m::Tensor& output() override {
    return a_;
  };

  // Assumes Y is one hos encoded
  float loss(m::Tensor& X, m::Tensor& Y) override {
  }

  static std::shared_ptr<Layer> newLayer(std::size_t batch_size, std::size_t input_dim) {
    return std::make_shared<ReluLayer>(batch_size, input_dim);
  }

private:
  m::Tensor a_;                 // Activation value  
};

class SoftmaxLayer : public Layer {
 public:
  SoftmaxLayer(std::size_t batch_size,
	    std::size_t input_dim) : a_{batch_size, input_dim} {}
  
  virtual ~SoftmaxLayer() {};

  // Not implemented for loss functions
  void forward(m::Tensor& X) override {
    std::vector<float> exp(X.cols());
    for (auto offset = 0 ; offset < X.rows() * X.cols() ; offset += X.cols()) {
      const auto init = X.data() + offset;
      const auto end = init + X.cols();
      const auto max_value = *std::max_element(init, end);
      std::transform(init, end, std::begin(exp),
                     [max_value](const float a) -> auto {
                       return std::exp(a * (1 / max_value) - 1);
                     });
      const auto sum = std::accumulate(exp.begin(), exp.end(), 0.0f);
      std::transform(exp.begin(), exp.end(), a_.data() + offset,
                     [sum](float a) -> auto { return a * (1 / sum); });
    }
  }

  // grad_in: Tensor. Labels
  // in: Tensor. the output of the network
  // grad_out: Tensor. The gradient resulting from this layer
  void backward(m::Tensor& grad_in, m::Tensor& in, m::Tensor& grad_out) override {}

  m::Tensor& output() override {
    return a_;
  };

  // Assumes Y is one hos encoded
  float loss(m::Tensor& X, m::Tensor& Y) override {
  }

  static std::shared_ptr<Layer> newLayer(std::size_t batch_size, std::size_t input_dim) {
    return std::make_shared<SoftmaxLayer>(batch_size, input_dim);
  }

private:
  m::Tensor a_;                 // Activation value  
};


class DenseLayer : public Layer {
 public:
  /*
    X matrix has each datapoint in a row <batch_size, input_dim>
    W matrix stores the weights for each unit in columnts <input, output>
    B is a row vector added to each row in the result of X * W
    Z = X * W + B

    Note: We can probably tuck the biases in the w_ matrix by adding an extra row
   */
  DenseLayer(std::size_t batch_size,
             std::size_t input_dim,
             std::size_t output_dim) : w_{input_dim, output_dim},
                                       b_{1, output_dim},
                                       z_{batch_size, output_dim}  {}
  

  virtual ~DenseLayer() {}

  void forward(m::Tensor& X) override {
    // z_ = X * w_
    m::matrix_mult<float>(X, w_, z_);

    // Add the bias to each row
    for (auto row = 0 ; row < z_.rows() ; ++row) {
      auto init = z_.data() + row * z_.cols();
      // z_i = z_i + bias
      std::transform(init, init +  z_.cols(), b_.data(), init,
                     [](float a, float b) { return a + b; });
    }

    std::cout << z_ << std::endl;
  }

  void backward(m::Tensor& grad_in, m::Tensor& in, m::Tensor& grad_out) override {
  }

  m::Tensor& output() override {
    return z_;
  }

  static std::shared_ptr<Layer> newLayer(std::size_t batch_size,
                                         std::size_t input_dim,
                                         std::size_t output_dim) {
    return std::make_shared<DenseLayer>(batch_size,input_dim, output_dim);
  }

 private:
  m::Tensor w_;                 // Weights
  m::Tensor b_;                 // Bias
  m::Tensor z_;                 // Value after linear combination
};


class NN {
 public:
  NN(std::vector<std::shared_ptr<Layer>>&& layers) : layers_shared_{std::move(layers)}{}
  void addLoss(std::shared_ptr<Layer> loss) {
    loss_ = loss;
  }

  void forward(m::Tensor& X) {
    m::Tensor* input = &X;
    for (auto& layer : layers_shared_) {
      layer->forward(*input);
      input = &layer->output();
    }
  }

  void fit(m::Tensor& X, m::Tensor& Y) {
    forward(X);

    // Gradient after going through a layer or loss function
    // Initialized with the dimensions of the network output
    auto grad_out = std::make_unique<m::Tensor>(Y.rows(), Y.cols());
    if (loss_) {
      loss_->backward(Y, layers_shared_.back()->output(), *grad_out);
    } else {
      return;
    }


    // Back propagate through all the layers
    for (auto it = layers_shared_.rbegin() ; it != layers_shared_.rend() ; ++it) {
      std::unique_ptr<m::Tensor> grad_in;
      std::swap(grad_in, grad_out);

      // For all layers except the first
      if ((it + 1) != layers_shared_.rend()) {
        auto& prev_layer_output = (*(it + 1))->output();
        grad_out = std::make_unique<m::Tensor>(prev_layer_output.rows(), prev_layer_output.cols());
        (*it)->backward(*grad_in, prev_layer_output, *grad_out);
      } else {
        // The first layer
        grad_out = std::make_unique<m::Tensor>(X.rows(), X.cols());
        (*it)->backward(*grad_in, X, *grad_out);
      }
    }


  }
 private:
  std::vector<std::shared_ptr<Layer>> layers_shared_;
  std::shared_ptr<Layer> loss_;
};

} // namespace nn
