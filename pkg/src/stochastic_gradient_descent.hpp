#ifndef _sgd_STOCHASTIC_GRADIENT_DESCENT_H
#define _sgd_STOCHASTIC_GRADIENT_DESCENT_H

#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

#include "gradient.hpp"
#include "activation.hpp"
#include "cross_entropy.hpp"

Rcpp::NumericVector stochastic_gradient_descent(Rcpp::NumericMatrix X, 
						Rcpp::NumericMatrix y, 
						int epochs, 
						double learning_rate, 
						double momentum, 
						int minibatch_size,
						double l2_reg,
						bool shuffle,
						int verbosity);

#endif
