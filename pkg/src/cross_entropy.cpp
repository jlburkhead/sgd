#include "cross_entropy.hpp"

using namespace Rcpp;

arma::mat cross_entropy(NumericMatrix y, NumericMatrix h) {
  
  arma::mat ym = as< arma::mat >(y);
  arma::mat hm = as< arma::mat >(h);
  
  return -(ym % log(hm) + (1 - ym) % log(1 - hm));

}
