#ifndef _sgd_ACTIVATION_H
#define _sgd_ACTIVATION_H

#include <RcppArmadillo.h>

#include "sigmoid.hpp"

Rcpp::NumericVector activation(Rcpp::NumericMatrix X, Rcpp::NumericVector w);

#endif
