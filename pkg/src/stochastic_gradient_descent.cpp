#include "stochastic_gradient_descent.hpp"


using namespace Rcpp;

void stochastic_gradient_descent(NumericMatrix X, 
				 NumericMatrix y,
				 arma::mat& w,
				 arma::mat (*act) (arma::mat, arma::mat),
				 int epochs, 
				 double learning_rate, 
				 double momentum,
				 int minibatch_size = 100,
				 double l2_reg = 0.0,
				 bool shuffle = true) {
  
  RNGScope scope;
  int n = X.nrow(); // number of observations
  int p = X.ncol(); // dimensionality
  int k = y.ncol(); // number of classes
  if (minibatch_size == 0)
    minibatch_size = n;
  int minibatches = (n - 1) / minibatch_size + 1; // number of minibatches

  // TODO: initialize weights more better
  arma::mat Xm = as< arma::mat >(X);
  arma::mat ym = as< arma::mat >(y);
  if (w.n_cols == 0)
    w = arma::randn(p, k);
  arma::mat delta_w = arma::zeros(p, k);
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
      
      arma::mat h = (*act)(X_minibatch, w);
      arma::mat g = gradient(X_minibatch, y_minibatch, h) + lambda % w; 

      delta_w = momentum * delta_w + (1 - momentum) * learning_rate * g;
      w = w - delta_w;

    }
     
  }

}

