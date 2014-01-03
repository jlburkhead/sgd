#include "sigmoid.hpp"

using namespace Rcpp;

NumericMatrix sigmoid(NumericMatrix x) {
  arma::mat xm = as< arma::mat >(x);
  return wrap(1 / (1 + exp(-xm)));
}
