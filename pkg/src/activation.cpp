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

arma::mat softmax_activation(arma::mat X, arma::mat w) {
  arma::mat unnorm = exp(X * w);
  row_normalize(unnorm);
  return unnorm;
}
    
