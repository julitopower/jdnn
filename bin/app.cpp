#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "nn.hpp"

struct HyperParameters {
  float lr;
  std::size_t epochs;
};

HyperParameters readConfig(int argc, char** argv) {
  if (argc < 3)  {
    return HyperParameters{.lr = 0.01, .epochs = 10};
  } else {
    auto lr = std::stof(argv[1]);
    std::size_t epochs = std::stoi(argv[2]);
    return HyperParameters{.lr = lr, .epochs = epochs};
  }
}

void generate_data(std::size_t degree, std::size_t train_datapoints, std::size_t test_datapoints,
                   std::vector<float>& coef,
                   std::vector<float>& X_train, std::vector<float>& Y_train,
                   std::vector<float>& X_test, std::vector<float>& Y_test, std::size_t seed = 42) {
  // Initialize the RNG
  std::default_random_engine generator;
  if (seed == 42) {
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  } else {
    generator.seed(seed);
  }
  std::uniform_real_distribution<float> distribution{-1.0, 1.0};

  // Randomly initialize the values
  X_train.resize(degree * train_datapoints);
  Y_train.resize(train_datapoints);
  std::transform(X_train.begin(), X_train.end(), X_train.begin(),
                 [&distribution, &generator](float) {return distribution(generator);});



  // Calculate the labels
  std::vector<float> tmp(degree);
  for (auto i = 0U ; i < train_datapoints; ++i) {
    // Calculate the X^i terms of the equation
    std::size_t exp = 0;    
    std::transform(X_train.begin() + i * degree, X_train.begin() + (i + 1) * degree, coef.begin(),
                   tmp.begin(),
                   [&exp](float x, float w) {
                            return std::pow(x, exp++) * w;
                          });
    
    // Calculate the label by applying the weights to the X^i terms and adding bias
    const float acc = std::accumulate(tmp.begin(), tmp.end(), coef.back());
    Y_train[i] = acc;
  }

  // Initialize test dataset
  // Randomly initialize the values
  X_test.resize(degree * test_datapoints);
  Y_test.resize(test_datapoints);
  std::transform(X_test.begin(), X_test.end(), X_test.begin(),
                 [&distribution, &generator](float) {return distribution(generator);});


  // Calculate the labels
  for (auto i = 0U ; i < test_datapoints; ++i) {
    // Calculate the X^i terms of the equation
    std::size_t exp = 0;    
    std::transform(X_test.begin() + i * degree, X_test.begin() + (i + 1) * degree, coef.begin(),
                   tmp.begin(),
                   [&exp](float x, float w) {
                            return std::pow(x, exp++) * w;
                          });

    // Calculate the label by applying the weights to the X^i terms and adding bias
    const float acc = std::accumulate(tmp.begin(), tmp.end(), coef.back());
    Y_test[i] = acc;
  }
}

int main(int argc, char** argv) {

  auto hp = readConfig(argc, argv);
  
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
  std::size_t batch_size = 64;
  std::size_t input_dim = 3;
  std::size_t output_dim = 1;
  nn::NN nn{{nn::DenseLayer::newLayer(batch_size, input_dim, 3),
             nn::ReluLayer::newLayer(batch_size, 3),

             nn::DenseLayer::newLayer(batch_size, 3, 5),
             nn::ReluLayer::newLayer(batch_size, 5),
             
             nn::DenseLayer::newLayer(batch_size, 5, output_dim)}};
 
  nn.addLoss(nn::MSE::newLayer());
  nn::lr = hp.lr;
  
  m::Matrix<float> X{batch_size, input_dim};
  m::Matrix<float> Y{batch_size, output_dim};  
  std::vector<float> data;
  std::vector<float> labels;

  m::Matrix<float> X_test{batch_size, input_dim};
  m::Matrix<float> Y_test{batch_size, output_dim};  
  std::vector<float> data_test;
  std::vector<float> labels_test;  

  // Randomly initialize coefficients
  std::default_random_engine generator;
  if (true) {
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  } else {
    generator.seed(42);
  }
  std::uniform_real_distribution<float> distribution{-1.0, 1.0};  
  std::vector<float> coef(input_dim + 1);
  std::transform(coef.begin(), coef.end(), coef.begin(),
                 [&distribution, &generator](float ) {return distribution(generator);});

  const auto train_size = 800;
  const auto test_size = 20000;
  generate_data(input_dim, train_size, test_size, coef, data, labels, data_test, labels_test);

  for (auto epoch = 0 ; epoch < hp.epochs ; ++epoch) {
    for (auto batch = 0 ; batch < train_size / batch_size ; ++batch) {
      X.data(data.data() + batch * batch_size * X.cols());
      Y.data(labels.data() + batch * batch_size * Y.cols());
      nn.fit(X, Y);
    }
    nn.loss(X, Y);
    std::cout << "Train Loss: " << nn.loss(X, Y) / batch_size;

    float test_loss = 0.0f;
    for (auto batch = 0 ; batch < test_size / batch_size ; ++batch) {
      X_test.data(data_test.data() + batch * batch_size * X_test.cols());
      Y_test.data(labels_test.data() + batch * batch_size * Y_test.cols());
      test_loss += nn.loss(X_test, Y_test);
    }
    std::cout << " Test Loss: " << test_loss / test_size << std::endl;
  }
}
