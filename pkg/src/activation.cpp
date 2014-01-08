#include "activation.hpp"

arma::mat activation(arma::mat X, arma::mat w) {
  return X * w;
}

arma::mat sigmoid_activation(arma::mat X, arma::mat w) {
  return sigmoid(activation(X, w));
}



