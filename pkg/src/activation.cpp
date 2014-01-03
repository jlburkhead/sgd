#include "activation.hpp"

arma::mat activation(arma::mat X, arma::mat w) {
  return sigmoid(X * w);
}


