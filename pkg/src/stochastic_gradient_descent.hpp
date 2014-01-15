#ifndef _sgd_STOCHASTIC_GRADIENT_DESCENT_H
#define _sgd_STOCHASTIC_GRADIENT_DESCENT_H

#include <RcppArmadillo.h>

#include "gradient.hpp"
#include "arma_utils.hpp"
#include "multilayer_perceptron.hpp"

void stochastic_gradient_descent(Rcpp::NumericMatrix X, 
				 Rcpp::NumericMatrix y, 
				 arma::mat& w,
				 arma::mat (*act) (arma::mat, arma::mat),
				 int epochs, 
				 double learning_rate, 
				 double momentum, 
				 int minibatch_size,
				 double l2_reg,
				 bool shuffle);

void mlp_gradient_descent(Rcpp::NumericMatrix X,
			  Rcpp::NumericMatrix y,
			  logistic_layer& ll,
			  softmax_layer& sl,
			  int epochs,
			  double learning_rate,
			  double momentum,
			  int minibatch_size,
			  double l2_reg,
			  bool shuffle);

#endif
