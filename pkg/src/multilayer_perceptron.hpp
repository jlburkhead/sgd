#ifndef _sgd_MULTILAYER_PERCEPTRON_H
#define _sgd_MULTILAYER_PERCEPTRON_H

#include <RcppArmadillo.h>

#include "activation.hpp"
#include "sigmoid.hpp"

class base_layer {
public:
  base_layer(bool use_bias_, 
	     arma::mat (*act_) (arma::mat, arma::mat),
	     arma::mat (*derivative_) (arma::mat));

  arma::mat forward_propagate(arma::mat X);
  arma::mat backpropagate(arma::mat delta, double learning_rate, double momentum);
  arma::mat activate(arma::mat X);
  arma::mat coef();
    

private:
  void init_weight_(int n_in, int n_out);

  int size;
  bool use_bias;
  arma::mat w, a, bias, incoming_activation, delta_w, delta_bias;
  arma::mat (*act) (arma::mat, arma::mat);
  arma::mat (*derivative) (arma::mat);
};

class logistic_layer : public base_layer {
public:
  logistic_layer();
};

class softmax_layer : public base_layer {
public:
  softmax_layer();
};

class mlp {
public:
  mlp(Rcpp::List l);

  void fit(Rcpp::NumericMatrix X, Rcpp::NumericMatrix y);
  Rcpp::NumericMatrix predict(Rcpp::NumericMatrix X);
  Rcpp::List coef();

private:
  bool shuffle;
  int n_hidden, epochs, minibatch_size;
  double learning_rate, momentum, l2_reg;
  logistic_layer hidden;
  softmax_layer output;
};

#endif
