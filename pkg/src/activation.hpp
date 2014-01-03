#ifndef _sgd_ACTIVATION_H
#define _sgd_ACTIVATION_H

#include <RcppArmadillo.h>

#include "sigmoid.hpp"

Rcpp::NumericMatrix activation(Rcpp::NumericMatrix X, arma::mat w);

#endif
