#ifndef _sgd_SIGMOID_H
#define _sgd_SIGMOID_H

#include <RcppArmadillo.h>

arma::mat sigmoid(arma::mat x);
arma::mat d_sigmoid(arma::mat x);

#endif
