#include "shuffle_matrix.hpp"

using namespace Rcpp;

NumericMatrix shuffle_matrix(Rcpp::NumericMatrix A) {
  
  IntegerVector index = seq_len(A.nrow());
  IntegerVector order = RcppArmadillo::sample(index, 
					      index.size(), 
					      false, 
					      NumericVector::create());
  NumericMatrix out(A.nrow(), A.ncol());
  
  for (int i = 0; i < order.size(); i++)
    out(i, _) = A(order[i] - 1, _);

  return out;

}

