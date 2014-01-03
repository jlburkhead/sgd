#include "stochastic_gradient_descent.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector stochastic_gradient_descent(NumericMatrix X, 
					  NumericVector y, 
					  int max_epoch, 
					  double learning_rate, 
					  double momentum, 
					  bool shuffle = true,
					  bool verbose = false, 
					  double tol = 1.0e-7) {
  
  int n = X.nrow(); // number of observations
  int p = X.ncol(); // dimensionality

  // TODO: initialize weights more better
  NumericVector w = rnorm(p);
  double last_l = R_PosInf;
  NumericVector delta_w(p);
  
  for (int e = 0; e < max_epoch; e++) {
    
    double l = 0;
    
    if (shuffle) {
      IntegerVector index = seq_len(n);
      IntegerVector order = RcppArmadillo::sample(index,
						  n,
						  false,
						  NumericVector::create());
      NumericMatrix X_shuffled(n, p);
      NumericVector y_shuffled(n);
      
      for (int i = 0; i < n; i++) {
	X_shuffled(i, _) = X(order[i] - 1, _);
	y_shuffled(i) = y[order[i] - 1];
      }
      
      X = X_shuffled;
      y = y_shuffled;
    } 
    

    for (int i = 0; i < n; i++) {
      NumericMatrix row(1, p, X(i, _).begin());
            
      NumericVector h = activation(row, w);
      NumericVector g = gradient(row,
				 h,
				 NumericVector::create(y[i]) // slicing vector returns a double
				 ); 

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

