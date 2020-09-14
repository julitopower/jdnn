#pragma once

#include <gsl/gsl_blas.h>
#include <ostream>
#include <cstring>
#include <random>
#include <chrono>
#include <functional>
#include <algorithm>

namespace m {
template<typename T> 
class Matrix {
 public:
  Matrix(std::size_t nrows, std::size_t ncols) :
      data_{new T[nrows * ncols]}, nrows_{nrows}, ncols_{ncols} {
        // By default initialize the matrix with random values
        std::default_random_engine generator;
        generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::uniform_real_distribution<float> distribution{-1.0, 1.0};
        for(auto i = 0 ; i < nrows * ncols ; ++i) {
          data_[i] = distribution(generator);
        }
      }

  Matrix(const Matrix& m) = delete;
  ~Matrix() {
    delete[] data_;
  }

  T* data() { return data_; }
  const T* cdata() const { return data_; }

  void data(const T*d) {
    std::memcpy(data_, d, sizeof(T) * ncols_ * nrows_);
  }

  void apply(std::function<float(float)> f, Matrix& out) {
    std::transform(data_, data_ + nrows_ * ncols_, out.data(), f);
  }
  
  std::size_t rows() { return nrows_; }
  std::size_t cols() { return ncols_; }
  
  template <typename K>
  friend std::ostream &operator<<(std::ostream& os, const Matrix<K>& m);
  
 private:
  T* data_;
  const std::size_t nrows_;
  const std::size_t ncols_;
};

template <typename T>
std::ostream &operator<<(std::ostream& os, const Matrix<T>& m) {
  for (auto i = 0U ; i < m.nrows_ ; ++i) {
    for (auto j = 0U ; j < m.ncols_ ; ++j) {
      os << m.data_[i * m.ncols_ + j] << ", ";
    }
    os << std::endl;
  }
  return os;
}

using TransTy = CBLAS_TRANSPOSE_t;
constexpr auto Trans = CblasTrans;
constexpr auto NoTrans = CblasNoTrans;

template<typename T>
void matrix_mult(Matrix<T>& op1, TransTy trans_op1,
                 Matrix<T>& op2, TransTy trans_op2,
                 Matrix<T>& dest) {
  auto gsl_op1 = gsl_matrix_float_view_array(op1.data(), op1.rows(), op1.cols()).matrix;
  auto gsl_op2 = gsl_matrix_float_view_array(op2.data(), op2.rows(), op2.cols()).matrix;
  auto gsl_dest = gsl_matrix_float_view_array(dest.data(), dest.rows(),dest.cols()).matrix;    
  gsl_blas_sgemm(trans_op1, trans_op2,
                 1.0, &gsl_op1, &gsl_op2,
                 0.0, &gsl_dest);
}

template<typename T>
void matrix_mult(Matrix<T>& op1, Matrix<T>& op2, Matrix<T>& dest) {
  matrix_mult(op1, NoTrans, op2, NoTrans, dest);
}

using Tensor = Matrix<float>;

} // namespace m
