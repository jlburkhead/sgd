#include "stochastic_gradient_descent.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector stochastic_gradient_descent(NumericMatrix X, 
					  NumericVector y, 
					  int max_epoch, 
					  double learning_rate, 
					  double momentum, 
					  bool verbose = false, 
					  double tol = 1.0e-7) {
  
  int n = X.nrow(); // number of observations
  int p = X.ncol(); // dimensionality

  // TODO: initialize weights more better
  NumericVector w = rnorm(p);
  double last_l = R_PosInf;
  NumericVector delta_w(p);
  arma::mat Xm = as< arma::mat >(X);
  
  
  for (int e = 0; e < max_epoch; e++) {
    
    double l = 0;
    
    // TODO: add shuffling

    for (int i = 0; i < n; i++) {
      NumericMatrix row = wrap(Xm(arma::span(i, i), arma::span(0, p - 1)));
      
      NumericVector h = activation(row, w);
      NumericVector g = gradient_double(row, h[0], y[i]);
      delta_w = momentum * delta_w + (1 - momentum) * learning_rate * g;
      w = w - delta_w;
      double ce = cross_entropy< double >(y[i], h[0]);
      l += ce;
    }
     
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

