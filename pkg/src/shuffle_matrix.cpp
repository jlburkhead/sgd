#include "shuffle_matrix.hpp"

using namespace Rcpp;

void shuffle_matrix(arma::mat& A, arma::mat& B) {
  
  IntegerVector index = seq_len(A.n_rows);
  IntegerVector order = RcppArmadillo::sample(index, 
					      index.size(), 
					      false, 
					      NumericVector::create());
  arma::mat newA(A.n_rows, A.n_cols);
  arma::mat newB(B.n_rows, B.n_cols);
  
  for (int i = 0; i < order.size(); i++) {
    newA(i, arma::span::all) = A(order[i] - 1, arma::span::all);
    newB(i, arma::span::all) = B(order[i] - 1, arma::span::all);
  }

  A = newA;
  B = newB;

}

