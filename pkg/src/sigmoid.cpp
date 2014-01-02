#include "sigmoid.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector sigmoid(NumericVector x) {
  return 1 / (1 + exp(-x));
}
