#ifndef _sgd_ACTIVATION_H
#define _sgd_ACTIVATION_H

#include <RcppArmadillo.h>

#include "sigmoid.hpp"
#include "arma_utils.hpp"

arma::mat activation(arma::mat X, arma::mat w);
arma::mat sigmoid_activation(arma::mat X, arma::mat w);
arma::mat poisson_activation(arma::mat X, arma::mat w);
arma::mat softmax_activation(arma::mat X, arma::mat w);

#endif
