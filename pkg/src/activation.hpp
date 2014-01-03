#ifndef _sgd_ACTIVATION_H
#define _sgd_ACTIVATION_H

#include <RcppArmadillo.h>

#include "sigmoid.hpp"

arma::mat activation(arma::mat X, arma::mat w);

#endif
