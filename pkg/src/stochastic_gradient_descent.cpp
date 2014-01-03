#include "stochastic_gradient_descent.hpp"

using namespace Rcpp;

// [[Rcpp::export]]
NumericVector stochastic_gradient_descent(NumericMatrix X, 
					  NumericVector y, 
					  int max_epoch, 
					  double learning_rate, 
					  double momentum,
					  int minibatch_size = 100,
					  bool shuffle = true,
					  bool verbose = false, 
					  double tol = 1.0e-7) {
  
  int n = X.nrow(); // number of observations
  int p = X.ncol(); // dimensionality
  int minibatches = n / minibatch_size; // number of minibatches

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
    


    for (int i = 0; i < minibatches; i++) {
      
      // make minibatch range
      int start = i * minibatch_size;
      int end = start + minibatch_size - 1;
      if (end > n - 1)
	end = n - 1;
      Range r(start, end);
      
      NumericMatrix row(minibatch_size, 
			p, 
			X(r, _).begin());

      NumericVector h = activation(row, w);
      NumericVector g = gradient(row,
				 h,
				 y[r]
				 ); 

      delta_w = momentum * delta_w + (1 - momentum) * learning_rate * g;
      w = w - delta_w;

      NumericVector ce = cross_entropy< NumericVector >(y[r], h);
      for (int j = 0; j < ce.size(); j++)
      	l += ce[j];
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

