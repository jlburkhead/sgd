#include <RcppArmadillo.h>

#include "activation.hpp"
#include "sigmoid.hpp"

using namespace Rcpp;


class base_layer {
public:
  base_layer(bool use_bias_, 
	     arma::mat (*act_) (arma::mat, arma::mat), 
	     arma::mat (*derivative_) (arma::mat)) :  use_bias(use_bias_),
						      act(act_),
						      derivative(derivative_) {}
  
  arma::mat forward_propagate(arma::mat X) {
    if (w.n_cols == 0) 
      init_weight_(X.n_cols, size);
    a = activate_(X);
    return a;
  }

  arma::mat backpropagate(arma::mat delta, 
			  double learning_rate,
			  double momentum) {
    delta_w = momentum * delta_w + (1 - momentum) * learning_rate * incoming_activation.t() * delta;
    w = w - delta_w;
    if (use_bias) {
      delta_bias = momentum * delta_bias + (1 - momentum) * learning_rate * arma::sum(delta, 0);
      bias = bias - delta_bias;
    }
    return delta * w.t() % derivative(incoming_activation);
  }
  
  arma::mat activate_(arma::mat X) {
    incoming_activation = X;
    arma::mat activation_(X.n_cols, size);
    if (use_bias) {
      X = arma::join_horiz(arma::ones(X.n_rows, 1), X);
      arma::mat wb = arma::join_vert(bias, w);
      activation_ = act(X, wb);
    } else {
      activation_ = act(X, w);
    }
    return activation_;
  }
  
  void init_weight_(int n_in, int n_out) {
    if (w.n_cols == 0) {
      size = n_out;
      w = arma::randn(n_in, n_out);
      delta_w = arma::zeros(n_in, n_out);
      if (use_bias) {
	bias = arma::randn(1, n_out);
	delta_bias = arma::zeros(1, n_out);
      }
    }
  }
  
  arma::mat coef() {
    return arma::join_vert(bias, w);
  }

protected:
  int size;
  bool use_bias;
  arma::mat w, a, bias, incoming_activation, delta_w, delta_bias;
  arma::mat (*act) (arma::mat, arma::mat);
  arma::mat (*derivative) (arma::mat);

};

class logistic_layer : public base_layer {
public:
  logistic_layer() : base_layer(true, sigmoid_activation, d_sigmoid) {}
};

class softmax_layer : public base_layer {
public:
  softmax_layer() : base_layer(true, softmax_activation, d_sigmoid) {}
};


class mlp {
public:
  mlp(List l) : n_hidden(as<int>(l["n_hidden"])),
		epochs(as<int>(l["epochs"])),
		learning_rate(as<double>(l["learning_rate"])),
		momentum(as<double>(l["momentum"])) {}

  void fit(NumericMatrix X, NumericMatrix y) {
    int n = X.nrow();
    arma::mat Xm = as< arma::mat >(X);
    arma::mat ym = as< arma::mat >(y);
    
    hidden.init_weight_(Xm.n_cols, n_hidden);
    output.init_weight_(n_hidden, ym.n_cols);
    
    for (int e = 0; e < epochs; e++) {
      arma::mat hidden_activation = hidden.forward_propagate(Xm);
      arma::mat output_activation = output.forward_propagate(hidden_activation);
      
      arma::mat output_delta = output_activation - ym;
      arma::mat hidden_delta = output.backpropagate(output_delta, 
						    learning_rate,
						    momentum);
      hidden.backpropagate(hidden_delta, learning_rate, momentum);
    }

  }
  
  NumericMatrix predict(NumericMatrix X) {
    arma::mat Xm = as< arma::mat >(X);
    return wrap(output.activate_(hidden.activate_(Xm)));
  }
  
  List coef() {
    return List::create(Named("hidden") = wrap(hidden.coef()),
			Named("output") = wrap(output.coef()));
  }

private:
  int n_hidden, epochs;
  double learning_rate, momentum;
  logistic_layer hidden;
  softmax_layer output;
};
  

RCPP_MODULE(MLP) {

  class_<mlp>(".mlp")
    .constructor<List>()
    
    .method("Fit", &mlp::fit)
    .method("Predict", &mlp::predict)
    .method("Coef", &mlp::coef)
    ;

}
