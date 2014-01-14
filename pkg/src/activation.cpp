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
  if (unnorm.n_cols == 1)
    return sigmoid(X * w); // return sigmoid in the binary case
  row_normalize(unnorm);
  return unnorm;
}
    
