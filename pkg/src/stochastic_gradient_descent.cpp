#include "stochastic_gradient_descent.hpp"

using namespace Rcpp;

// TODO: change this ridiculously long name

// [[Rcpp::export]]
NumericVector stochastic_gradient_descent(NumericMatrix X, 
					  NumericMatrix y, 
					  int epochs, 
					  double learning_rate, 
					  double momentum,
					  int minibatch_size = 100,
					  double l2_reg = 0.0,
					  bool shuffle = true,
					  int verbosity = 0) {
  
  int n = X.nrow(); // number of observations
  int p = X.ncol(); // dimensionality
  int k = y.ncol(); // number of classes
  int minibatches = (n - 1) / minibatch_size + 1; // number of minibatches
  Rcout.precision(10); // precision for print statements

  // TODO: initialize weights more better
  arma::mat Xm = as< arma::mat >(X);
  arma::mat ym = as< arma::mat >(y);
  arma::mat w = arma::randn(p, k);
  arma::mat delta_w(p, k);
  delta_w.fill(0);
  arma::mat lambda(p, k);
  lambda.fill(l2_reg);
  lambda(0, 0) = 0; // don't penalize bias
    
  for (int e = 0; e < epochs; e++) {
    
    double l = 0;
    
    if (shuffle && minibatch_size != n)
      shuffle_matrix(Xm, ym);
    
    for (int i = 0; i < minibatches; i++) {

      // make minibatch span
      int start = i * minibatch_size;
      int end = start + minibatch_size - 1;
      if (end > n - 1)
	end = n - 1;
      arma::span s(start, end);
      
      arma::mat X_minibatch = Xm(s, arma::span::all);
      arma::mat y_minibatch = ym(s, arma::span::all);
      
      arma::mat h = activation(X_minibatch, w);
      arma::mat g = gradient(X_minibatch, y_minibatch, h) + lambda % w; 

      if (verbosity >= 2)
	for (int j = 0; j < p; j++)
	  Rcout << std::fixed << w[j] << std::endl;

      delta_w = momentum * delta_w + (1 - momentum) * learning_rate * g;
      w = w - delta_w;

      arma::mat ce = cross_entropy(y_minibatch, h);
      l += arma::accu(ce);

    }
     
    if (verbosity >= 1)
      Rcout << "epoch " << e + 1 << " cross-entropy:\t" << std::fixed << l << std::endl;

  }

  return wrap(w);

}

