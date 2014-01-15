#include "activation.hpp"

arma::mat identity(arma::mat X) {
  return X;
}

arma::mat activation(arma::mat X, arma::mat w) {
  return X * w;
}

arma::mat sigmoid_activation(arma::mat X, arma::mat w) {
  return sigmoid(X * w);
}

arma::mat exponential_activation(arma::mat X, arma::mat w) {
  return exp(X * w);
}

arma::mat softmax(arma::mat X) {
  arma::mat unnorm = exp(X);
  row_normalize(unnorm);
  return unnorm;
}
    
