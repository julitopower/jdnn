#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "nn.hpp"

int main(int argc, char** argv) {

  if (false) {
    std::cout << "Simple matrix multiplication" << std::endl;    
    const std::size_t nrows = 3;
    const std::size_t ncols = 3;
    m::Matrix<float> m1{nrows, ncols};
    m1.data(std::vector<float>({1,2,3,1,2,3,1,2,3}).data());
    m::Matrix<float> m2{nrows, ncols};
    m2.data(std::vector<float>({1,2,3,1,2,3,1,2,3}).data());  
    m::Matrix<float> m3{nrows, ncols};
    m::matrix_mult(m1, m2, m3);
    std::cout << m1 << m2 << m3 << std::endl;
  }

  if (false) {
    m::Matrix<float> m{3,3};
    m::Matrix<float> m2{3,3};
    m::Matrix<float> m3{3,3};
    m::matrix_mult(m, m2, m3);
    std::cout << m << m2 << m3 << std::endl;
  }

  // NN testing
  std::size_t batch_size = 5;
  std::size_t input_dim = 3;
  std::size_t output_dim = 4;
  nn::NN nn{{nn::DenseLayer::newLayer(batch_size, input_dim, 2, nn::Relu),
             nn::DenseLayer::newLayer(batch_size, 2, 5, nn::Relu),
             nn::DenseLayer::newLayer(batch_size, 5, output_dim, nn::Softmax)}};
  nn.addLoss(nn::CrossEntropy::newLayer());
  m::Matrix<float> X{batch_size, input_dim};
  X.data(std::vector<float>{1,1,1,2,2,2,3,3,3}.data());

  m::Matrix<float> Y{batch_size, output_dim};
  Y.data(std::vector<float>{
          1,0,0,0,
          0,1,0,0,
          0,0,1,0
          }.data());  
  
  nn.fit(X, Y);
  
}
