#ifndef _sgd_GRADIENT_H
#define _sgd_GRADIENT_H

#include <RcppArmadillo.h>

arma::mat gradient(arma::mat X, arma::mat y, arma::mat h);

#endif
