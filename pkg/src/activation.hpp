#ifndef _sgd_ACTIVATION_H
#define _sgd_ACTIVATION_H

#include <RcppArmadillo.h>

#include "sigmoid.hpp"
#include "arma_utils.hpp"

arma::mat identity(arma::mat X);
arma::mat activation(arma::mat X, arma::mat w);
arma::mat sigmoid_activation(arma::mat X, arma::mat w);
arma::mat exponential_activation(arma::mat X, arma::mat w);
arma::mat softmax(arma::mat X);

#endif
