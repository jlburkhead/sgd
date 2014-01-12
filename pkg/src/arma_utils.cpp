#include "arma_utils.hpp"

using namespace Rcpp;

void row_normalize(arma::mat& X) {
  for (int i = 0; i < X.n_rows; i++)
    X.row(i) = X.row(i) / arma::accu(X.row(i));
}

IntegerVector row_max(arma::mat X) {
  int n = X.n_rows;
  IntegerVector out(n);

  for (int i = 0; i < n; i++) {
    arma::mat row = X(i, arma::span::all);
    out[i] = which_max(row);
  }

  return out;

}
  
int which_max(arma::mat X) {
  arma::uword index;
  X.max(index);
  return index;
}

void shuffle_matrix(arma::mat& A, arma::mat& B) {
  
  RNGScope scope;
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

