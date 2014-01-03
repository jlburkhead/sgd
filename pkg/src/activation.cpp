#include "activation.hpp"

using namespace Rcpp;

NumericMatrix activation(NumericMatrix X, arma::mat w) {
  
  arma::mat Xm = as< arma::mat >(X);
  
  return sigmoid(wrap(Xm * w));
}


