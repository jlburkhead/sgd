#include "gradient_descent.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector gradient_descent(NumericMatrix X, 
			       NumericVector y, 
			       int max_epoch, 
			       double learning_rate, 
			       double momentum, 
			       bool verbose = false, 
			       double tol = 1.0e-7) {
  
  // TODO: initialize weights more better
  NumericVector w = rnorm(X.ncol());
  double last_l = R_PosInf;
  NumericVector delta_w(X.ncol());

  for (int e = 0; e < max_epoch; e++) {
    NumericVector h = activation(X, w);
    NumericVector g = gradient(X, h, y);
    delta_w = momentum * delta_w + (1 - momentum) * learning_rate * g;
    w = w - delta_w;
    NumericVector ce = cross_entropy< NumericVector >(y, h);
    double l = 0;
    for (int i = 0; i < ce.size(); i++)
      l += ce[i];
    
    if (verbose) {
      Rcout.precision(10);
      Rcout << "epoch " << e + 1 << " cross-entropy:\t" << std::fixed << l << std::endl;
    }

    if (std::abs(last_l - l) < tol && e != 0)
      break;

    last_l = l;
  }

  return w;

}

  
