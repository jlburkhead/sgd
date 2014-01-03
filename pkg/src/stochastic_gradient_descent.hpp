#ifndef _sgd_STOCHASTIC_GRADIENT_DESCENT_H
#define _sgd_STOCHASTIC_GRADIENT_DESCENT_H

#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

#include "gradient.hpp"
#include "activation.hpp"
#include "cross_entropy.hpp"

Rcpp::NumericVector stochastic_gradient_descent(Rcpp::NumericMatrix X, 
						Rcpp::NumericVector y, 
						int max_epoch, 
						double learning_rate, 
						double momentum, 
						bool shuffle,
						bool verbose, 
						double tol);

#endif
