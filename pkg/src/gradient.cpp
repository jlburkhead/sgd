#include "gradient.hpp"

using namespace Rcpp;

// TODO: figure out how to combine these

NumericVector gradient(NumericMatrix X, NumericVector h, NumericVector y) {

  arma::mat Xm = as< arma::mat >(X);
  arma::colvec hm = as< arma::colvec >(h);
  arma::colvec ym = as< arma::colvec >(y);
  
  return wrap( Xm.t() * (hm - ym) );
  
}

NumericVector gradient_double(NumericMatrix X, double h, double y) {

  arma::mat Xm = as< arma::mat >(X);

  return wrap( Xm.t() * (h - y) );

}
