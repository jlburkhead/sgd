#include "gradient.hpp"

using namespace Rcpp;

arma::mat gradient(NumericMatrix X, NumericMatrix h, NumericMatrix y) {

  arma::mat Xm = as< arma::mat >(X);
  arma::mat hm = as< arma::mat >(h);
  arma::mat ym = as< arma::mat >(y);
  
  return Xm.t() * (hm - ym);
  
}
