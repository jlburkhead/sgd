#ifndef _sgd_MULTILAYER_PERCEPTRON_H
#define _sgd_MULTILAYER_PERCEPTRON_H

#include <RcppArmadillo.h>

#include "activation.hpp"
#include "sigmoid.hpp"

class base_mlp {
public:
  base_mlp(Rcpp::List l);

protected:
  void init_fit_();
  void init_param_();
  void fit_(arma::mat X, arma::mat y);
  arma::mat predict_(arma::mat X);
  
  int n_hidden, n_features, n_outputs, epochs, t, minibatch_size;
  double learning_rate, momentum;
  bool shuffle;
  arma::mat w1, w2, b1, b2, hidden_activation;
  arma::mat delta_w1, delta_w2, delta_b1, delta_b2;
  arma::mat (*hidden_func) (arma::mat);
  arma::mat (*hidden_derivative) (arma::mat);
  arma::mat (*output_func) (arma::mat);
};

class mlp_classifier : public base_mlp {
public:
  mlp_classifier(Rcpp::List l);

  void fit(Rcpp::NumericMatrix X, Rcpp::NumericMatrix y);
  Rcpp::NumericMatrix predict(Rcpp::NumericMatrix X);
  // Rcpp::NumericMatrix predict_class(Rcpp::NumericMatrix X);
};


#endif
