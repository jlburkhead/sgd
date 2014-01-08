#ifndef _sgd_STOCHASTIC_GRADIENT_DESCENT_H
#define _sgd_STOCHASTIC_GRADIENT_DESCENT_H

#include <RcppArmadillo.h>

#include "gradient.hpp"
#include "shuffle_matrix.hpp"

void stochastic_gradient_descent(Rcpp::NumericMatrix X, 
				 Rcpp::NumericMatrix y, 
				 arma::mat& w,
				 arma::mat (*act) (arma::mat, arma::mat),
				 int epochs, 
				 double learning_rate, 
				 double momentum, 
				 int minibatch_size,
				 double l2_reg,
				 bool shuffle,
				 int verbosity);

#endif
