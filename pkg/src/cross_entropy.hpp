#ifndef _sgd_CROSS_ENTROPY_H
#define _sgd_CROSS_ENTROPY_H

#include <RcppArmadillo.h>

arma::mat cross_entropy(Rcpp::NumericMatrix y, Rcpp::NumericMatrix h);

#endif
