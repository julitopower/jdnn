#include <iostream>
#include <vector>

#include "matrix.hpp"
#include "nn.hpp"

struct HyperParameters {
  float lr;
  std::size_t epochs;
  std::size_t batch_size;
};

HyperParameters readConfig(int argc, char** argv) {
  if (argc < 4)  {
    return HyperParameters{.lr = 0.00001, .epochs = 10, .batch_size = 32};
  } else {
    auto lr = std::stof(argv[1]);
    std::size_t epochs = std::stoi(argv[2]);
    std::size_t batch_size = std::stoi(argv[3]);
    return HyperParameters{.lr = lr, .epochs = epochs, .batch_size = batch_size};
  }
}

void generate_data(std::size_t degree, std::size_t train_datapoints, std::size_t test_datapoints,
                   std::vector<float>& X_train, std::vector<float>& Y_train,
                   std::vector<float>& X_test, std::vector<float>& Y_test, std::size_t seed = 42) {
  // Initialize the RNG
  std::default_random_engine generator;
  if (seed == 42) {
    generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  } else {
    generator.seed(seed);
  }
  std::uniform_real_distribution<float> distribution{-0.5, 1.0};

  // initialize coeeficients
  std::vector<float> coef(degree + 1);
  std::transform(coef.begin(), coef.end(), coef.begin(),
                 [&distribution, &generator](float ) {return distribution(generator);});  
  
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

  // Load hyperparameters
  auto hp = readConfig(argc, argv);

  // NN testing
  std::size_t input_dim = 3;
  std::size_t output_dim = 1;
  nn::NN nn{{nn::DenseLayer::newLayer(input_dim, 3),
             nn::ReluLayer::newLayer(3),

             nn::DenseLayer::newLayer(3, 5),
             nn::ReluLayer::newLayer(5),
             
             nn::DenseLayer::newLayer(5, output_dim)},
            hp.batch_size};
 
  nn.addLoss(nn::MSE::newLayer());
  nn::lr = hp.lr;
  
  m::Matrix<float> X{hp.batch_size, input_dim};
  m::Matrix<float> Y{hp.batch_size, output_dim};  
  std::vector<float> data;
  std::vector<float> labels;

  m::Matrix<float> X_test{hp.batch_size, input_dim};
  m::Matrix<float> Y_test{hp.batch_size, output_dim};  
  std::vector<float> data_test;
  std::vector<float> labels_test;

  const auto train_size = 80000;
  const auto test_size = 20000;
  generate_data(input_dim, train_size, test_size, data, labels, data_test, labels_test, 123);

  for (auto epoch = 0U ; epoch < hp.epochs ; ++epoch) {
    for (auto batch = 0U ; batch < train_size / hp.batch_size ; ++batch) {
      X.data(data.data() + batch * hp.batch_size * X.cols());
      Y.data(labels.data() + batch * hp.batch_size * Y.cols());
      nn.fit(X, Y);
    }



    float test_loss = 0.0f;
    for (auto batch = 0U ; batch < test_size / hp.batch_size ; ++batch) {
      const auto row_offset = batch * hp.batch_size;
      X_test.data(data_test.data() +  row_offset * X_test.cols());
      Y_test.data(labels_test.data() + row_offset * Y_test.cols());
      test_loss += nn.loss(X_test, Y_test);
    }

    if (epoch % 100 == 0) {
      nn.loss(X, Y);
      std::cout << "Train Loss: " << nn.loss(X, Y) / hp.batch_size;
      std::cout << " Test Loss: " << test_loss / test_size << std::endl;      
    }    

  }
}
