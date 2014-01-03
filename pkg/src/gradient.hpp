#ifndef _sgd_GRADIENT_H
#define _sgd_GRADIENT_H

#include <RcppArmadillo.h>

arma::mat gradient(Rcpp::NumericMatrix X, Rcpp::NumericMatrix h, Rcpp::NumericMatrix y);

#endif
