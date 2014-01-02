#include "cross_entropy.hpp"

using namespace Rcpp;

template <typename T>
T cross_entropy(T y, T h) {
  return -(y * log(h) + (1 - y) * log(1 - h));
}

template double cross_entropy< double >(double y, double h);
template NumericVector cross_entropy< NumericVector >(NumericVector y, NumericVector h);
