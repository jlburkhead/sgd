#include "stochastic_gradient_descent.hpp"

using namespace Rcpp;

// TODO: change this ridiculously long name

// [[Rcpp::export]]
NumericVector stochastic_gradient_descent(NumericMatrix X, 
					  NumericMatrix y, 
					  int max_epoch, 
					  double learning_rate, 
					  double momentum,
					  int minibatch_size = 100,
					  bool shuffle = true,
					  int verbosity = 0,
					  double tol = 1.0e-7) {
  
  int n = X.nrow(); // number of observations
  int p = X.ncol(); // dimensionality
  int k = y.ncol(); // number of classes
  int minibatches = n / minibatch_size; // number of minibatches
  Rcout.precision(10); // precision for print statements

  // TODO: initialize weights more better
  arma::mat w = arma::randn(p, k);
  arma::mat delta_w(p, k);
  delta_w.fill(0);
  double last_l = R_PosInf;
    
  for (int e = 0; e < max_epoch; e++) {
    
    double l = 0;
    
    // TODO: put this back into a shuffle_matrix function
    if (shuffle) {
      IntegerVector index = seq_len(n);
      IntegerVector order = RcppArmadillo::sample(index,
						  n,
						  false,
						  NumericVector::create());
      NumericMatrix X_shuffled(n, p);
      NumericMatrix y_shuffled(n, k);
      
      for (int i = 0; i < n; i++) {
	X_shuffled(i, _) = X(order[i] - 1, _);
	y_shuffled(i, _) = y(order[i] - 1, _);
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
      
      arma::mat X_minibatch = as< arma::mat >(wrap(X(r, _)));
      arma::mat y_minibatch = as< arma::mat >(wrap(y(r, _)));
      
      // only X and y use Rcpp::NumericMatrix
      // everything else is arma::mat
      arma::mat h = activation(X_minibatch, w);
      arma::mat g = gradient(X_minibatch, y_minibatch, h); 

      if (verbosity >= 2)
	for (int j = 0; j < p; j++)
	  Rcout << std::fixed << w[j] << std::endl;

      delta_w = momentum * delta_w + (1 - momentum) * learning_rate * g;
      w = w - delta_w;

      arma::mat ce = cross_entropy(y_minibatch, h);
      l += arma::accu(ce);

    }
     
    if (verbosity >= 1) {
      Rcout << "epoch " << e + 1 << " cross-entropy:\t" << std::fixed << l << std::endl;
    }
    
    // TODO: tol should be some proportion of last loss not an absolute level
    if (std::abs(last_l - l) < tol && e != 0)
      break;
    
    last_l = l;

  }

  return wrap(w);

}

