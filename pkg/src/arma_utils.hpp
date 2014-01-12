#ifndef _sgd_ARMA_UTILS_H
#define _sgd_ARMA_UTILS_H

#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

void row_normalize(arma::mat& X);
Rcpp::IntegerVector row_max(const arma::mat X);
int which_max(arma::mat X);
void shuffle_matrix(arma::mat& A, arma::mat& B);

#endif
