#include <cstdarg>
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>

#include "matrix.hpp"

namespace nn {

using Shape = std::vector<std::size_t>;
using ActivationFunction = std::function<void(m::Tensor&, m::Tensor&)>;

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
    std::size_t idx = 0;
    float loss = 0.0;
    for (auto offset = 0 ; offset < Y.rows() * Y.cols() ; offset += Y.cols()) {
      // Find the index of the class from the ground truth
      const auto begin = Y.data() + offset;
      idx = std::find(begin, begin + Y.cols(), 1.0) - begin;

      // Accumulate the loss for the batch
      loss -= log2f(X.data()[offset + idx]);
    }
    return loss;
  }

  static std::shared_ptr<Layer> newLayer() {
    return std::make_shared<CrossEntropy>();
  }
};

// TODO: This needs to be reworked. Activation functions need to be able to
// be part of the back propagation, for which we need their derivatives
// Relu activation function
auto Relu = [](m::Tensor& in, m::Tensor& out) -> auto {
              std::transform(in.data(), in.data() + in.rows() * in.cols(), out.data(),
                             [](float a) { return std::max(0.0f, a); });
            };

auto Softmax = [](m::Tensor& in, m::Tensor& out) -> auto {
                 std::vector<float> exp(in.cols());
                 for (auto offset = 0 ; offset < in.rows() * in.cols() ; offset += in.cols()) {
                   std::transform(in.data() + offset, in.data() + offset + in.cols(), std::begin(exp),
                                  [](float a) -> auto {
                                    return std::exp(a);
                                  });
                   const auto sum = std::accumulate(exp.begin(), exp.end(), 0.0f);
                   std::transform(exp.begin(), exp.end(), out.data() + offset,
                                  [sum](float a) -> auto {
                                    return a * 1/sum;
                                  });
                 }
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
             std::size_t output_dim,
             bool activation_function = false) : w_{input_dim, output_dim},
                                                b_{1, output_dim},
                                                z_{batch_size, output_dim},
                                                a_{batch_size, output_dim},
                                                activation_function_{activation_function} {}

  DenseLayer(std::size_t batch_size,
             std::size_t input_dim,
             std::size_t output_dim,
             ActivationFunction activation_function) :
      DenseLayer(batch_size, input_dim, output_dim, true) {
    afn_ = activation_function;
  }

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
    // Apply relu by default
    if (activation_function_) {
      afn_(z_, a_);
    }
    std::cout << a_ << std::endl;
  }

  void backward(m::Tensor& grad_in, m::Tensor& in, m::Tensor& grad_out) override {
    std::cout << "Backward through dense layer" << std::endl;
  }

  m::Tensor& output() override {
    return a_;
  }

  static std::shared_ptr<Layer> newLayer(std::size_t batch_size,
                                         std::size_t input_dim,
                                         std::size_t output_dim) {
    return std::make_shared<DenseLayer>(batch_size,input_dim, output_dim, false);
  }

  static std::shared_ptr<Layer> newLayer(std::size_t batch_size,
                                         std::size_t input_dim,
                                         std::size_t output_dim,
                                         ActivationFunction activation_function) {
    return std::make_shared<DenseLayer>(batch_size, input_dim, output_dim, activation_function);
  }

 private:
  m::Tensor w_;                 // Weights
  m::Tensor b_;                 // Bias
  m::Tensor z_;                 // Value after linear combination
  m::Tensor a_;                 // Activation value
  bool activation_function_;    // Wheter there is an activation function for this layer
  ActivationFunction afn_;      // Activation function
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
      std::cout << "Loss: " << loss_->loss(layers_shared_.back()->output(), Y) << std::endl;
      loss_->backward(Y, layers_shared_.back()->output(), *grad_out);
      std::cout << *grad_out << std::endl;
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
      std::cout << *grad_out << std::endl;
    }


  }
 private:
  std::vector<std::shared_ptr<Layer>> layers_shared_;
  std::shared_ptr<Layer> loss_;
};

} // namespace nn
