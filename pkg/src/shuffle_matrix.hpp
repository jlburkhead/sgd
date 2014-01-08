#ifndef _sgd_SHUFFLE_H
#define _sgd_SHUFFLE_H

#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>

void shuffle_matrix(arma::mat& A, arma::mat& B);
void print_mat(arma::mat A, arma::mat B);

#endif
