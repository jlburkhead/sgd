#include "activation.hpp"

arma::mat activation(arma::mat X, arma::mat w) {
  return X * w;
}

arma::mat sigmoid_activation(arma::mat X, arma::mat w) {
  return sigmoid(X * w);
}

arma::mat poisson_activation(arma::mat X, arma::mat w) {
  return exp(X * w);
}

