#include "activation.hpp"

using namespace Rcpp;

NumericVector activation(NumericMatrix X, NumericVector w) {
  
  arma::mat Xm = as< arma::mat >(X);
  arma::colvec wm = as< arma::colvec >(w);
  
  return sigmoid(wrap(Xm * wm));

}


