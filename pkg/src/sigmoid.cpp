#include "sigmoid.hpp"

arma::mat sigmoid(arma::mat x) {
  return 1 / (1 + exp(-x));
}

arma::mat d_sigmoid(arma::mat x) {
  return sigmoid(x) % (1 - sigmoid(x));
}
