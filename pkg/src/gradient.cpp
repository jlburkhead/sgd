#include "gradient.hpp"

using namespace Rcpp;

NumericVector gradient(NumericMatrix X, NumericVector h, NumericVector y) {

  arma::mat Xm = as< arma::mat >(X);
  arma::colvec hm = as< arma::colvec >(h);
  arma::colvec ym = as< arma::colvec >(y);
  
  return wrap( Xm.t() * (hm - ym) );
  
}
