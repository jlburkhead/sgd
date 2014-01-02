#ifndef _sgd_GRADIENT_H
#define _sgd_GRADIENT_H

#include <RcppArmadillo.h>

Rcpp::NumericVector gradient(Rcpp::NumericMatrix X, Rcpp::NumericVector h, Rcpp::NumericVector y);
Rcpp::NumericVector gradient_double(Rcpp::NumericMatrix X, double h, double y);

#endif
