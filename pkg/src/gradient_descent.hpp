#ifndef _sgd_GRADIENT_DESCENT_H
#define _sgd_GRADIENT_DESCENT_H

#include <RcppArmadillo.h>

#include "gradient.hpp"
#include "activation.hpp"
#include "cross_entropy.hpp"

Rcpp::NumericVector gradient_descent(Rcpp::NumericMatrix X, Rcpp::NumericVector y, int max_epoch, double learning_rate, double momentum, double tol);

#endif
